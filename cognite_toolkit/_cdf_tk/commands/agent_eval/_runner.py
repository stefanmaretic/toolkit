"""Runs eval cases against an agent: chats with the agent turn by turn and scores the answers.

The chat loop, the parsing of agent responses and the pass/fail rules match `cognite agents eval`
in @cognite/cli.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rich.console import Console

from cognite_toolkit._cdf_tk.client import ToolkitClient
from cognite_toolkit._cdf_tk.client.http_client import ToolkitAPIError
from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import (
    AgentChatRequest,
    AgentChatResponse,
    AgentResponseMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatContextInformation,
    ChatTextContent,
    ToolConfirmationChatMessage,
    UserChatMessage,
    UserSessionChatMessage,
)
from cognite_toolkit._cdf_tk.client.resource_classes.session import (
    OneshotTokenExchangeSessionRequest,
    SessionCreateRequest,
    TokenExchangeSessionRequest,
)

from ._errors import AgentEvalError
from ._suite import EvalCase, EvalTool

DEFAULT_MAX_TURNS = 8
TOOL_TEXT_LIMIT = 500
ChatRoute = Literal["public", "internal"]


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()


class EvalLogger(Protocol):
    def log(self, message: str) -> None: ...

    def log_json(self, label: str, value: Any, max_chars: int = 6000) -> None: ...


class ConsoleEvalLogger:
    """Writes request/response traces to stderr (`--verbose`)."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True, markup=False, highlight=False, soft_wrap=True)

    def log(self, message: str) -> None:
        self._console.print(message, markup=False, highlight=False)

    def log_json(self, label: str, value: Any, max_chars: int = 6000) -> None:
        self.log(f"{label}:\n{format_json(value, max_chars)}")


class TraceCapture:
    """Keeps the verbose traces in memory so that they can be written to the eval report."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(self, message: str) -> None:
        self.events.append({"type": "log", "message": message})

    def log_json(self, label: str, value: Any, max_chars: int = 6000) -> None:
        self.events.append({"type": "json", "label": label, "value": _clone_for_trace(value)})


class TeeEvalLogger:
    def __init__(self, *loggers: EvalLogger) -> None:
        self._loggers = loggers

    def log(self, message: str) -> None:
        for logger in self._loggers:
            logger.log(message)

    def log_json(self, label: str, value: Any, max_chars: int = 6000) -> None:
        for logger in self._loggers:
            logger.log_json(label, value, max_chars)


def format_json(value: Any, max_chars: int = 6000) -> str:
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n… ({len(text) - max_chars} more chars)"


def _clone_for_trace(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def format_duration_ms(milliseconds: float) -> str:
    if milliseconds < 1000:
        return f"{round(milliseconds)}ms"
    return f"{milliseconds / 1000:.1f}s"


def format_duration_log(label: str, milliseconds: float) -> str:
    return f"{label} finished in {format_duration_ms(milliseconds)}"


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _elapsed_ms(start: float) -> int:
    return max(0, round(_now_ms() - start))


class EvalChatService(Protocol):
    """The calls the runner and the judge make. `EvalService` implements them against CDF."""

    def post_agent_chat(self, request: AgentChatRequest) -> AgentChatResponse: ...

    def post_chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    def create_session_nonce(self) -> str: ...


class EvalService:
    """The CDF calls made during an eval: agent chat, chat completions (the judge) and sessions."""

    def __init__(self, client: ToolkitClient, route: ChatRoute = "public", chat_base_url: str | None = None) -> None:
        self.client = client
        self.route = route
        self.chat_base_url = chat_base_url
        self.agent_chat_url = client.agents.chat_url(internal=route == "internal", base_url=chat_base_url)
        self.chat_completions_url = client.chat_completions.url

    def post_agent_chat(self, request: AgentChatRequest) -> AgentChatResponse:
        try:
            return self.client.agents.chat(request, internal=self.route == "internal", base_url=self.chat_base_url)
        except ToolkitAPIError as error:
            if self.chat_base_url is not None and error.code is None:
                raise AgentEvalError(
                    f"Agent chat failed: could not connect to {self.chat_base_url}",
                    hint=f"Is cog-ai running? Try GET {self.chat_base_url}/health",
                ) from error
            raise wrap_api_error(error, "Agent chat", self.agent_chat_url) from error

    def post_chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            return self.client.chat_completions.create(request)
        except ToolkitAPIError as error:
            raise wrap_api_error(error, "Chat completions", self.chat_completions_url) from error

    def create_session_nonce(self) -> str:
        """Create a session for a `userSession` action. Falls back to token exchange if one-shot is rejected."""
        try:
            return self._request_session_nonce(OneshotTokenExchangeSessionRequest())
        except AgentEvalError as error:
            if error.http_status in (400, 401, 403):
                return self._request_session_nonce(TokenExchangeSessionRequest())
            raise

    def _request_session_nonce(self, item: SessionCreateRequest) -> str:
        url = self.client.config.create_api_url("/sessions")
        try:
            created = self.client.sessions.create([item])
        except ToolkitAPIError as error:
            raise wrap_api_error(error, "Session creation", url) from error
        nonce = created[0].nonce if created else None
        if not nonce:
            raise AgentEvalError(
                "Session creation returned no nonce",
                hint="The Sessions API returned an empty items list. Retry, or check IdP configuration for this project.",
            )
        return nonce


def wrap_api_error(error: ToolkitAPIError, label: str, url: str) -> AgentEvalError:
    detail = error.response.error.message if error.response is not None else error.message
    body: Any = None
    if error.response is not None:
        try:
            body = json.loads(error.response.body)
        except ValueError:
            body = error.response.body
    suffix = f": {detail}" if detail else ""
    return AgentEvalError(
        f"{label} failed ({error.code}){suffix}", http_status=error.code, request_url=url, response_body=body
    )


@dataclass
class ToolCall:
    name: str
    input: Any = UNSET
    result: Any = UNSET
    synthesized: bool = False

    def dump(self) -> dict[str, Any]:
        output: dict[str, Any] = {"name": self.name}
        if self.input is not UNSET:
            output["input"] = self.input
        if self.result is not UNSET:
            output["result"] = self.result
        return output


@dataclass
class AgentTurn:
    output: str
    tool_calls: list[ToolCall]
    cursor: str | None
    pending_tool_confirmation: tuple[str, str | None] | None = None
    pending_user_session: str | None = None


@dataclass
class AgentInvocation:
    output: str
    tool_calls: list[ToolCall]
    round_trips: int
    cursor: str | None


@dataclass
class TurnRun:
    input: str
    output: str
    tool_calls: list[ToolCall]
    api_round_trips: int


@dataclass
class CaseRun:
    case_id: str
    turns: list[TurnRun]


@dataclass
class ScoreResult:
    case_id: str
    turn_index: int
    scorer_type: str
    score: int | float
    passed: bool
    label: str | None = None
    explanation: str | None = None


@dataclass
class CaseResult:
    eval_case: EvalCase
    scores: list[ScoreResult]
    passed: bool
    chat_duration_ms: int
    duration_ms: int
    run: CaseRun | None = None
    invoke_error: Exception | None = None
    score_error: Exception | None = None
    judge_duration_ms: int | None = None


@dataclass
class EvalSummary:
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    results: list[CaseResult] = field(default_factory=list)


def case_verdict_passed(result: CaseResult) -> bool:
    """Whether a case met expectations. For `expectPass: false`, the scorers are expected to fail."""
    if result.invoke_error is not None or result.score_error is not None:
        return False
    return result.passed if result.eval_case.expect_pass else not result.passed


def count_verdict_passes(results: Sequence[CaseResult]) -> int:
    return sum(1 for result in results if case_verdict_passed(result))


def count_verdict_failures(results: Sequence[CaseResult]) -> int:
    return sum(1 for result in results if not case_verdict_passed(result))


def is_unexpected_pass(result: CaseResult) -> bool:
    return (
        not result.eval_case.expect_pass
        and result.invoke_error is None
        and result.score_error is None
        and result.passed
    )


def count_unexpected_passes(results: Sequence[CaseResult]) -> int:
    return sum(1 for result in results if is_unexpected_pass(result))


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def compact_json(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def truncate(text: str, limit: float) -> str:
    if len(text) <= limit:
        return text
    cut = int(limit)
    return f"{text[:cut]}… ({len(text) - cut} more chars)"


def format_tool_invocation(call: ToolCall, limit: float = float("inf")) -> str:
    if call.input is UNSET or call.input is None:
        return f"{call.name}()"
    if isinstance(call.input, dict):
        if not call.input:
            return f"{call.name}()"
        arguments = ", ".join(f"{key}={compact_json(value)}" for key, value in call.input.items())
    else:
        arguments = compact_json(call.input)
    return f"{call.name}({truncate(arguments, limit)})"


def format_tool_result_preview(value: Any, limit: float = TOOL_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else compact_json(value)
    return truncate(" ".join(text.split()), limit)


def format_tool_names(calls: Sequence[ToolCall | EvalTool]) -> str:
    return ", ".join(call.name for call in calls)


def message_text(message: AgentResponseMessage) -> str:
    if message.content is not None and isinstance(message.content.text, str) and message.content.text:
        return message.content.text
    parts = [
        content.text
        for reasoning in message.reasoning or []
        for content in reasoning.content or []
        if isinstance(content.text, str) and content.text
    ]
    return "\n".join(parts).strip()


def _extract_tool_calls(messages: Sequence[AgentResponseMessage]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for message in messages:
        for reasoning in message.reasoning or []:
            for data in reasoning.data or []:
                call = data.tool_call
                if call is None or not isinstance(call.name, str) or not call.name:
                    continue
                calls.append(
                    ToolCall(
                        name=call.name,
                        input=call.input if "input" in call.model_fields_set else UNSET,
                        result=call.result if "result" in call.model_fields_set else UNSET,
                    )
                )
    return calls


def _has_reasoning_tool_call(message: AgentResponseMessage) -> bool:
    return any(
        data.tool_call is not None and isinstance(data.tool_call.name, str) and bool(data.tool_call.name)
        for reasoning in message.reasoning or []
        for data in reasoning.data or []
    )


def _synthesize_tool_confirmation_calls(messages: Sequence[AgentResponseMessage]) -> list[ToolCall]:
    """Tool calls the agent asked to confirm, for messages that do not report the tool call in their reasoning."""
    calls: list[ToolCall] = []
    for message in messages:
        if _has_reasoning_tool_call(message):
            continue
        for action in message.actions or []:
            confirmation = action.tool_confirmation
            if confirmation is None or not isinstance(confirmation.tool_name, str) or not confirmation.tool_name:
                continue
            calls.append(
                ToolCall(
                    name=confirmation.tool_name,
                    input=confirmation.tool_arguments if "tool_arguments" in confirmation.model_fields_set else UNSET,
                    synthesized=True,
                )
            )
    return calls


def merge_tool_calls(existing: list[ToolCall], new: Sequence[ToolCall]) -> None:
    """Add new tool calls. A real call replaces the synthesized call for the same tool and input."""
    for call in new:
        if call.synthesized:
            existing.append(call)
            continue
        index = next(
            (
                no
                for no, current in enumerate(existing)
                if current.synthesized
                and current.name == call.name
                and stable_json(current.input) == stable_json(call.input)
            ),
            None,
        )
        if index is None:
            existing.append(call)
        else:
            existing[index] = call


def extract_agent_turn(response: AgentChatResponse) -> AgentTurn:
    result = response.response
    if result is None or result.type != "result" or result.messages is None:
        raise ValueError("Unexpected agent chat response shape")
    messages = result.messages
    if not messages:
        raise ValueError("Agent returned no messages")
    last = messages[-1]
    tool_calls = [*_extract_tool_calls(messages), *_synthesize_tool_confirmation_calls(messages)]
    output = message_text(last)

    confirmation = next((action for action in last.actions or [] if action.type == "toolConfirmation"), None)
    if confirmation is not None:
        action_id = confirmation.action_id or confirmation.id
        if not action_id:
            raise ValueError("Tool confirmation action missing actionId (both actionId and id are absent)")
        tool_name = confirmation.tool_confirmation.tool_name if confirmation.tool_confirmation else None
        return AgentTurn(output, tool_calls, result.cursor, pending_tool_confirmation=(action_id, tool_name))

    session = next((action for action in last.actions or [] if action.type == "userSession"), None)
    if session is not None:
        action_id = session.action_id or session.id
        if not action_id:
            raise ValueError("User session action missing actionId (both actionId and id are absent)")
        return AgentTurn(output, tool_calls, result.cursor, pending_user_session=action_id)

    return AgentTurn(output, tool_calls, result.cursor)


def _redact_nonces(body: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {**message, "nonce": "<redacted>"}
        if message.get("role") == "action" and message.get("type") == "userSession"
        else message
        for message in body.get("messages", [])
    ]
    return {**body, "messages": messages}


def invoke_agent_case(
    service: EvalChatService,
    agent_external_id: str,
    input: str,
    cursor: str | None = None,
    context_information: ChatContextInformation | None = None,
    allow_tool_confirmation: bool = True,
    logger: EvalLogger | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AgentInvocation:
    """Send one user message and answer tool confirmation and user session actions until the agent is done."""
    if logger:
        logger.log(f"agentExternalId={agent_external_id} | auto-allow tools={str(allow_tool_confirmation).lower()}")
    pending_confirmation: str | None = None
    pending_session: tuple[str, str] | None = None
    tool_calls: list[ToolCall] = []
    output = ""
    for round_trip in range(1, max_turns + 1):
        label = f"turn {round_trip}/{max_turns}"
        request: AgentChatRequest
        if pending_session is not None and cursor:
            action_id, nonce = pending_session
            request = AgentChatRequest(
                agent_external_id=agent_external_id,
                cursor=cursor,
                messages=[UserSessionChatMessage(action_id=action_id, nonce=nonce)],
            )
            pending_session = None
        elif pending_confirmation is not None and cursor:
            request = AgentChatRequest(
                agent_external_id=agent_external_id,
                cursor=cursor,
                messages=[ToolConfirmationChatMessage(action_id=pending_confirmation, status="ALLOW")],
            )
            pending_confirmation = None
        else:
            request = AgentChatRequest(
                agent_external_id=agent_external_id,
                messages=[UserChatMessage(content=ChatTextContent(text=input))],
            )
            if cursor:
                request.cursor = cursor
        if context_information is not None:
            request.context_information = context_information

        if logger:
            logger.log(f"{label}: sending chat request…")
            logger.log_json(f"{label} request body", _redact_nonces(request.dump()))
        response = service.post_agent_chat(request)
        if logger:
            logger.log_json(f"{label} response", response.dump())

        try:
            turn = extract_agent_turn(response)
        except ValueError as error:
            raise AgentEvalError(
                f"Failed to parse agent response on turn {round_trip}/{max_turns}",
                hint="The CDF agent API returned an unexpected response shape. Run with --verbose to see the raw response.",
            ) from error
        merge_tool_calls(tool_calls, turn.tool_calls)
        if logger:
            logger.log(
                f"{label}: extracted output length={len(turn.output)}, tools=[{format_tool_names(turn.tool_calls)}]"
            )
        if turn.output:
            output = turn.output

        if turn.pending_tool_confirmation is not None:
            action_id, tool_name = turn.pending_tool_confirmation
            if logger:
                logger.log(f"{label}: tool confirmation requested — tool={tool_name or '?'} actionId={action_id}")
            if not allow_tool_confirmation:
                raise AgentEvalError(
                    f"Agent requested tool confirmation ({tool_name or 'unknown'}) but allowToolConfirmation is false",
                    hint="Re-run with tool confirmation enabled, or adjust the agent to avoid confirmation prompts.",
                )
            if not turn.cursor:
                raise AgentEvalError(
                    "Tool confirmation missing cursor or actionId",
                    hint="The agent API returned an incomplete tool-confirmation action. This may be an API bug — "
                    "retry or report to CDF support.",
                )
            cursor = turn.cursor
            pending_confirmation = action_id
            if logger:
                logger.log(f"{label}: auto-ALLOW, continuing with cursor…")
            continue

        if turn.pending_user_session is not None:
            if logger:
                logger.log(f"{label}: userSession requested — actionId={turn.pending_user_session}, creating nonce…")
            if not turn.cursor:
                raise AgentEvalError(
                    "User session missing cursor or actionId",
                    hint="The agent API returned an incomplete user-session action. This may be an API bug — "
                    "retry or report to CDF support.",
                )
            nonce = service.create_session_nonce()
            cursor = turn.cursor
            pending_session = (turn.pending_user_session, nonce)
            if logger:
                logger.log(f"{label}: created nonce, continuing with cursor…")
            continue

        if logger:
            logger.log(f"{label}: conversation complete")
        return AgentInvocation(
            output=output, tool_calls=tool_calls, round_trips=round_trip, cursor=turn.cursor or cursor
        )

    raise AgentEvalError(
        f"Agent chat exceeded {max_turns} turns (action loop?)",
        hint="The agent may be stuck in a tool-confirmation or user-session loop. Increase maxTurns or inspect the "
        "agent configuration.",
    )


def _annotate_turn_failure(error: Exception, turn_index: int, turn_count: int) -> Exception:
    if turn_count <= 1:
        return error
    message = error.message if isinstance(error, AgentEvalError) else str(error)
    hint = error.hint if isinstance(error, AgentEvalError) else None
    annotated = AgentEvalError(f"turn {turn_index + 1}/{turn_count}: {message}", hint=hint)
    annotated.__cause__ = error
    return annotated


ScoreCase = Callable[[EvalCase, CaseRun], list[ScoreResult]]


def run_case(
    case: EvalCase,
    service: EvalChatService,
    agent_external_id: str,
    score_case: ScoreCase,
    allow_tool_confirmation: bool = True,
    logger: EvalLogger | None = None,
) -> CaseResult:
    if logger:
        logger.log(f'--- case "{case.id}" ({case.source}) invoke ---')
    chat_start = _now_ms()
    turns: list[TurnRun] = []
    cursor: str | None = None
    try:
        for turn_index, turn in enumerate(case.turns):
            if logger:
                logger.log(f'  turn {turn_index + 1}/{len(case.turns)}: "{turn.input}"')
            try:
                invocation = invoke_agent_case(
                    service,
                    agent_external_id,
                    turn.input,
                    cursor=cursor,
                    context_information=case.context_information,
                    allow_tool_confirmation=allow_tool_confirmation,
                    logger=logger,
                )
            except Exception as error:
                raise _annotate_turn_failure(error, turn_index, len(case.turns)) from error
            cursor = invocation.cursor
            turns.append(TurnRun(turn.input, invocation.output, invocation.tool_calls, invocation.round_trips))
    except Exception as error:
        elapsed = _elapsed_ms(chat_start)
        if logger:
            logger.log(format_duration_log(f'chat (case "{case.id}")', elapsed))
        return CaseResult(case, [], False, elapsed, elapsed, invoke_error=error)

    run = CaseRun(case.id, turns)
    chat_ms = _elapsed_ms(chat_start)
    if logger:
        logger.log(format_duration_log(f'chat (case "{case.id}")', chat_ms))
        logger.log(f'--- case "{case.id}" ({case.source}) score (Phoenix) ---')
    judge_start = _now_ms()
    try:
        scores = score_case(case, run)
    except Exception as error:
        judge_ms = _elapsed_ms(judge_start)
        if logger:
            logger.log(format_duration_log(f'judge (case "{case.id}")', judge_ms))
        return CaseResult(
            case, [], False, chat_ms, chat_ms + judge_ms, run=run, score_error=error, judge_duration_ms=judge_ms
        )
    judge_ms = _elapsed_ms(judge_start)
    if logger:
        logger.log(format_duration_log(f'judge (case "{case.id}")', judge_ms))
    passed = bool(scores) and all(score.passed for score in scores)
    return CaseResult(case, scores, passed, chat_ms, chat_ms + judge_ms, run=run, judge_duration_ms=judge_ms)


def run_eval(
    cases: Sequence[EvalCase],
    service: EvalChatService,
    agent_external_id: str,
    score_case: ScoreCase,
    allow_tool_confirmation: bool = True,
    fail_fast: bool = False,
    logger: EvalLogger | None = None,
) -> EvalSummary:
    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case, service, agent_external_id, score_case, allow_tool_confirmation, logger)
        results.append(result)
        if fail_fast and not case_verdict_passed(result):
            if logger:
                logger.log(f'--- fail-fast: stopping after failed case "{case.id}" ({case.source}) ---')
            break
    passed = sum(1 for result in results if result.passed)
    return EvalSummary(
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        skipped_cases=len(cases) - len(results),
        results=results,
    )
