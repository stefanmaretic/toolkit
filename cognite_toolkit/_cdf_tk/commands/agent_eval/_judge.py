"""The LLM judge that scores agent answers.

@cognite/cli scores with the Phoenix evaluators (@arizeai/phoenix-evals) and sends the judge prompt to CDF
`POST /ai/chat/completions`. This module sends the same request: a system message that asks for a JSON object
with `explanation` and `label`, followed by the Phoenix prompt template with the case's values filled in.
"""

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import ChatCompletionMessage, ChatCompletionRequest

from ._errors import AgentEvalError
from ._judge_templates import JUDGE_TEMPLATES, JudgeTemplate
from ._runner import (
    CaseRun,
    EvalChatService,
    EvalLogger,
    ScoreResult,
    ToolCall,
    format_duration_log,
    format_tool_invocation,
    format_tool_names,
)
from ._suite import (
    SCORER_TYPES,
    CorrectnessScorer,
    EvalCase,
    EvalTool,
    FaithfulnessScorer,
    Scorer,
    ToolInvocationScorer,
    ToolSelectionScorer,
)

DEFAULT_JUDGE_MODEL = "azure/gpt-5"
JUDGE_MODEL_ENV = "EVAL_JUDGE_MODEL"
PASS_THRESHOLD = 0.5
_PASS_FAIL_LABELS: dict[str, tuple[str, str]] = {
    "correctness": ("correct", "incorrect"),
    "faithfulness": ("faithful", "unfaithful"),
    "toolSelection": ("correct", "incorrect"),
    "toolInvocation": ("correct", "incorrect"),
}
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_MAX_PARALLEL_SCORERS = 8

WarnFn = Callable[[str, str | None], None]


def resolve_judge_model_id(cli_model_id: str | None = None, agent_model_fallback: str | None = None) -> str:
    """The judge model: --judge-model, then EVAL_JUDGE_MODEL, then the agent's model, then azure/gpt-5."""
    for candidate in (cli_model_id, os.environ.get(JUDGE_MODEL_ENV), agent_model_fallback):
        if candidate and candidate.strip():
            return candidate.strip()
    return DEFAULT_JUDGE_MODEL


def judge_model_source(cli_model_id: str | None, agent_model_fallback: str | None, agent_file: str) -> str:
    if cli_model_id:
        return "--judge-model flag"
    if os.environ.get(JUDGE_MODEL_ENV):
        return f"{JUDGE_MODEL_ENV} env"
    if agent_model_fallback:
        return f"{agent_file} model fallback"
    return "default"


def render_template(template: str, variables: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda match: variables.get(match.group(1), ""), template)


def _response_schema(labels: Sequence[str]) -> dict[str, Any]:
    # Same JSON Schema, with the same key order, as the Phoenix classifier sends through the AI SDK.
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"explanation": {"type": "string"}, "label": {"type": "string", "enum": list(labels)}},
        "required": ["explanation", "label"],
        "additionalProperties": False,
    }


def json_response_instruction(labels: Sequence[str]) -> str:
    schema = json.dumps(_response_schema(labels), separators=(",", ":"))
    return (
        "Respond with a single valid JSON object only. No markdown code fences or extra text.\n"
        f"The JSON must conform to this schema: {schema}"
    )


class CdfJudge:
    """Classifies an answer with a judge prompt template, using a CDF language model."""

    def __init__(self, model_id: str, service: EvalChatService, logger: EvalLogger | None = None) -> None:
        self.model_id = model_id
        self.service = service
        self.logger = logger

    def classify(self, template: JudgeTemplate, variables: dict[str, str]) -> dict[str, Any]:
        labels = list(template.choices)
        request = ChatCompletionRequest(
            model=self.model_id,
            messages=[
                ChatCompletionMessage(role="system", content=json_response_instruction(labels)),
                ChatCompletionMessage(role="user", content=render_template(template.template, variables)),
            ],
            stream=False,
        )
        if self.logger:
            self.logger.log_json("chat/completions request", request.dump())
            self.logger.log("judge POST /ai/chat/completions")
        start = _now_ms()
        response = self.service.post_chat_completions(request)
        if self.logger:
            self.logger.log(format_duration_log("judge /ai/chat/completions", _now_ms() - start))
            self.logger.log_json("chat/completions response", response.dump())
        label, explanation = _parse_classification(response.first_content(), labels)
        return {"score": template.choices[label], "label": label, "explanation": explanation}


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _parse_classification(content: str, labels: Sequence[str]) -> tuple[str, str]:
    try:
        parsed = json.loads(content)
    except ValueError:
        raise ValueError("No object generated: could not parse the response.") from None
    if (
        not isinstance(parsed, dict)
        or parsed.get("label") not in labels
        or not isinstance(parsed.get("explanation"), str)
    ):
        raise ValueError("No object generated: response did not match schema.")
    return parsed["label"], parsed["explanation"]


def scorer_passed(scorer_type: str, result: dict[str, Any]) -> bool:
    labels = _PASS_FAIL_LABELS.get(scorer_type)
    if labels is not None:
        if result.get("label") == labels[0]:
            return True
        if result.get("label") == labels[1]:
            return False
    score = result.get("score")
    return (score if isinstance(score, int | float) else 0) >= PASS_THRESHOLD


def conversation_context(run: CaseRun, turn_index: int) -> str:
    previous = run.turns[:turn_index]
    if not previous:
        return ""
    lines = "\n".join(f"User: {turn.input}\nAssistant: {turn.output}" for turn in previous)
    return f"Conversation so far:\n{lines}\n"


def format_available_tools(tools: Sequence[EvalTool]) -> str:
    return "\n".join(f"{tool.name}: {tool.description}" for tool in tools)


def _format_tool_selection(calls: Sequence[ToolCall]) -> str:
    return format_tool_names(calls) if calls else "No tools called"


def _format_tool_invocations(calls: Sequence[ToolCall]) -> str:
    return "\n".join(format_tool_invocation(call) for call in calls) if calls else "No tools called"


def _run_judge(
    judge: CdfJudge, case_id: str, turn_index: int, scorer_type: str, variables: dict[str, str]
) -> ScoreResult:
    logger = judge.logger
    if logger:
        logger.log(f"scoring {scorer_type} (Phoenix + CDF /ai/chat/completions)…")
        logger.log_json(f"{scorer_type} judge input", variables)
    start = _now_ms()
    try:
        result = judge.classify(JUDGE_TEMPLATES[scorer_type], variables)
    except AgentEvalError:
        raise
    except Exception as error:
        raise AgentEvalError(
            f"Phoenix {scorer_type} judge failed ({error}). Judge uses CDF POST .../ai/chat/completions."
        ) from error
    if logger:
        logger.log(format_duration_log(f"{scorer_type} judge", _now_ms() - start))
        logger.log_json(f"{scorer_type} judge result", result)
    passed = scorer_passed(scorer_type, result)
    score = result["score"] if isinstance(result.get("score"), int | float) else (1 if passed else 0)
    return ScoreResult(
        case_id=case_id,
        turn_index=turn_index,
        scorer_type=scorer_type,
        score=score,
        passed=passed,
        label=result.get("label"),
        explanation=result.get("explanation"),
    )


def evaluate_scorer(
    case: EvalCase,
    turn_index: int,
    scorer: Scorer,
    run: CaseRun,
    judge: CdfJudge,
    agent_tools: Sequence[EvalTool] | None = None,
    warn: WarnFn | None = None,
    agent_file_label: str = "*.agent.yaml",
) -> ScoreResult | None:
    """Score one scorer on one turn. Returns None if a tool scorer is skipped because no tools are declared."""
    if turn_index >= len(run.turns):
        raise AgentEvalError(
            f'No agent response recorded for turn {turn_index + 1} of case "{case.id}" ({case.source}).',
            hint="This is an internal mismatch between scored turns and executed turns.",
        )
    turn = run.turns[turn_index]
    context = conversation_context(run, turn_index)
    current_question = f"{context}Current question:\n{turn.input}" if context else turn.input

    if isinstance(scorer, CorrectnessScorer):
        variables = {
            "input": f"{context}User question:\n{turn.input}\nCriteria:\n{scorer.criteria}",
            "output": turn.output,
        }
        return _run_judge(judge, case.id, turn_index, scorer.type, variables)
    if isinstance(scorer, FaithfulnessScorer):
        variables = {"input": current_question, "output": turn.output, "context": scorer.ground_truth}
        return _run_judge(judge, case.id, turn_index, scorer.type, variables)
    if isinstance(scorer, ToolSelectionScorer | ToolInvocationScorer):
        tools = next((tools for tools in (scorer.tools, case.tools, agent_tools) if tools is not None), [])
        if not tools:
            if agent_tools is None:
                hint = (
                    "Add a tools field to the scorer, case, or eval.yaml root "
                    f"({agent_file_label} is not used with --external-id/--system-agent)."
                )
            else:
                hint = (
                    f"Add a tools field to the scorer, case, or eval.yaml root, or declare tools in {agent_file_label}."
                )
            message = f'Warning: Skipping {scorer.type} scorer on case "{case.id}": no tools available.'
            if judge.logger:
                judge.logger.log(message)
            if warn:
                warn(message, hint)
            return None
        selection = (
            _format_tool_selection(turn.tool_calls)
            if isinstance(scorer, ToolSelectionScorer)
            else _format_tool_invocations(turn.tool_calls)
        )
        variables = {
            "input": current_question,
            "availableTools": format_available_tools(tools),
            "toolSelection": selection,
        }
        return _run_judge(judge, case.id, turn_index, scorer.type, variables)

    supported = ", ".join(f'"{name}"' for name in SCORER_TYPES[:-1]) + f', and "{SCORER_TYPES[-1]}"'
    raise AgentEvalError(
        f'Scorer type "{getattr(scorer, "type", "?")}" is not supported yet.',
        hint=f"The CLI eval schema currently supports only {supported} scorers.",
    )


def make_case_scorer(
    judge: CdfJudge,
    agent_tools: Sequence[EvalTool] | None = None,
    warn: WarnFn | None = None,
    agent_file_label: str = "*.agent.yaml",
) -> Callable[[EvalCase, CaseRun], list[ScoreResult]]:
    """Scores all scorers of all turns of a case. The judge calls run in parallel; results keep their order."""

    def score_case(case: EvalCase, run: CaseRun) -> list[ScoreResult]:
        tasks = [(turn_index, scorer) for turn_index, turn in enumerate(case.turns) for scorer in turn.scorers or []]
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_SCORERS, len(tasks))) as executor:
            results = list(
                executor.map(
                    lambda task: evaluate_scorer(
                        case, task[0], task[1], run, judge, agent_tools, warn, agent_file_label
                    ),
                    tasks,
                )
            )
        return [result for result in results if result is not None]

    return score_case
