import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx2
import pytest
import respx
from rich.console import Console

from cognite_toolkit._cdf_tk.client import ToolkitClient, ToolkitClientConfig
from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import (
    AgentChatRequest,
    AgentChatResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from cognite_toolkit._cdf_tk.commands.agent_eval import AgentEvalCommand, AgentEvalError
from cognite_toolkit._cdf_tk.commands.agent_eval._judge import (
    CdfJudge,
    evaluate_scorer,
    make_case_scorer,
    resolve_judge_model_id,
    scorer_passed,
)
from cognite_toolkit._cdf_tk.commands.agent_eval._runner import CaseRun, TurnRun, invoke_agent_case
from cognite_toolkit._cdf_tk.commands.agent_eval._suite import EvalCase, filter_eval_cases, load_eval_cases


def write_files(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


SCORED_TURN = (
    "turns:\n      - input: Hi\n        scorers:\n          - type: correctness\n            criteria: Greets\n"
)


class TestLoadEvalCases:
    def test_cascade_of_session_context_tags_and_tools(self, tmp_path: Path) -> None:
        write_files(
            tmp_path,
            {
                "eval/eval.yaml": f"""include:
  - cases/maintenance.yaml
dataModels:
  - space: sp
    externalId: Model
    version: v1
instanceSpaces: [inst]
appContext: Plant A
tools:
  - name: find_assets
    description: Root tools
cases:
  - id: root-case
    {SCORED_TURN}""",
                "eval/cases/maintenance.yaml": f"""tags: [maintenance, ci]
appContext: Plant B
tools:
  - name: list_work_orders
    description: File tools
cases:
  - id: file-case
    tags: [ci, slow]
    additionalInstructions: Answer in Norwegian.
    {SCORED_TURN}""",
            },
        )

        root_case, file_case = load_eval_cases(tmp_path)

        assert root_case.source == "eval/eval.yaml"
        assert root_case.context_information is not None
        assert root_case.context_information.dump() == {
            "dataModels": [{"space": "sp", "externalId": "Model", "version": "v1"}],
            "instanceSpaces": ["inst"],
            "appContext": "Plant A",
        }
        assert [tool.name for tool in root_case.tools or []] == ["find_assets"]

        assert file_case.source == "eval/cases/maintenance.yaml"
        assert file_case.tags == ["maintenance", "ci", "slow"]
        assert file_case.context_information is not None
        assert file_case.context_information.app_context == "Plant B\n\nAnswer in Norwegian."
        assert file_case.context_information.instance_spaces == ["inst"]
        assert [tool.name for tool in file_case.tools or []] == ["list_work_orders"]

        assert [case.id for case in filter_eval_cases([root_case, file_case], tag_filter=["slow"])] == ["file-case"]

    def test_include_extension_published(self, tmp_path: Path) -> None:
        write_files(
            tmp_path,
            {
                "eval/eval.yaml": f"""includeExtension:
  path: ../agent-project-config.yaml
  type: published
cases:
  - id: with-extension
    appContext: Plant A
    {SCORED_TURN}""",
                "agent-project-config.yaml": """defaultAgent: null
systemAgentExtensions:
  - type: draft
    additionalInstructions: Draft only
  - type: published
    additionalInstructions: Be brief.
    additionalSkills: [pump-skill]
    additionalSubagents: [pump_agent]
""",
            },
        )

        (case,) = load_eval_cases(tmp_path)

        assert case.context_information is not None
        assert case.context_information.dump() == {
            "appContext": "Plant A\n\nBe brief.",
            "skills": ["pump-skill"],
            "subagents": ["pump_agent"],
        }
        assert case.published_subagents == ["pump_agent"]

    def test_finds_single_nested_eval_yaml(self, tmp_path: Path) -> None:
        write_files(tmp_path, {"suites/origem/eval.yaml": f"cases:\n  - id: nested\n    {SCORED_TURN}"})

        (case,) = load_eval_cases(tmp_path)

        assert case.source == "suites/origem/eval.yaml"

    @pytest.mark.parametrize(
        "files, message",
        [
            pytest.param(
                {"eval/eval.yaml": f"cases:\n  - id: a\n    {SCORED_TURN}  - id: a\n    {SCORED_TURN}"},
                'Duplicate case id "a"',
                id="duplicate id",
            ),
            pytest.param(
                {"eval/eval.yaml": "cases:\n  - id: a\n    turns:\n      - input: Hi\n"},
                'Case "a" (eval/eval.yaml) has no scored turns',
                id="no scored turn",
            ),
            pytest.param(
                {"eval/eval.yaml": f"instanceSpaces: [sp]\ncases:\n  - id: a\n    {SCORED_TURN}"},
                'Case "a" declares instanceSpaces without dataModels',
                id="instance spaces without data models",
            ),
            pytest.param(
                {
                    "eval/eval.yaml": "additionalSkills: [x]\nincludeExtension:\n  path: p.yaml\n  type: draft\n"
                    f"cases:\n  - id: a\n    {SCORED_TURN}"
                },
                "includeExtension cannot be combined with additionalInstructions",
                id="include extension is exclusive",
            ),
            pytest.param(
                {
                    "eval/eval.yaml": "cases:\n  - id: a\n    turns:\n      - input: Hi\n        scorers:\n          - type: x\n"
                },
                "cases.0.turns.0.scorers.0",
                id="unknown scorer type",
            ),
            pytest.param(
                {"eval.yaml": "cases: []\n", "eval/eval.yaml": "cases: []\n"},
                "Found both eval suite layouts",
                id="both layouts",
            ),
        ],
    )
    def test_invalid_suites(self, tmp_path: Path, files: dict[str, str], message: str) -> None:
        write_files(tmp_path, files)

        with pytest.raises(AgentEvalError) as exc_info:
            load_eval_cases(tmp_path)

        assert message in exc_info.value.message


def chat_response(text: str = "", cursor: str = "c1", actions: list[dict] | None = None, **message: Any) -> dict:
    return {
        "agentExternalId": "my_agent",
        "response": {
            "type": "result",
            "cursor": cursor,
            "messages": [
                {"role": "agent", "content": {"type": "text", "text": text}, "actions": actions or [], **message}
            ],
        },
    }


class FakeService:
    def __init__(self, chat: list[dict] | None = None, completions: list[str] | None = None) -> None:
        self.chat_requests: list[dict] = []
        self.completion_requests: list[ChatCompletionRequest] = []
        self._chat = list(chat or [])
        self._completions = list(completions or [])

    def post_agent_chat(self, request: AgentChatRequest) -> AgentChatResponse:
        self.chat_requests.append(request.dump())
        return AgentChatResponse.model_validate(self._chat.pop(0))

    def post_chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.completion_requests.append(request)
        return ChatCompletionResponse.model_validate({"choices": [{"message": {"content": self._completions.pop(0)}}]})

    def create_session_nonce(self) -> str:
        return "nonce-1"


class TestInvokeAgentCase:
    def test_answers_tool_confirmation_and_user_session(self) -> None:
        service = FakeService(
            chat=[
                chat_response(
                    cursor="c1",
                    actions=[
                        {
                            "type": "toolConfirmation",
                            "actionId": "a1",
                            "toolConfirmation": {"toolName": "execute", "toolArguments": {"code": "1+1"}},
                        }
                    ],
                ),
                chat_response(cursor="c2", actions=[{"type": "userSession", "id": "s1"}]),
                chat_response(
                    "Done",
                    cursor="c3",
                    reasoning=[{"data": [{"toolCall": {"name": "execute", "input": {"code": "1+1"}, "result": 2}}]}],
                ),
            ]
        )

        invocation = invoke_agent_case(service, "my_agent", "Compute", cursor="c0")

        assert [request["messages"] for request in service.chat_requests] == [
            [{"role": "user", "content": {"type": "text", "text": "Compute"}}],
            [{"role": "action", "type": "toolConfirmation", "actionId": "a1", "status": "ALLOW"}],
            [{"role": "action", "type": "userSession", "actionId": "s1", "nonce": "nonce-1"}],
        ]
        assert [request["cursor"] for request in service.chat_requests] == ["c0", "c1", "c2"]
        assert invocation.output == "Done"
        assert invocation.round_trips == 3
        assert invocation.cursor == "c3"
        # The synthesized call from the confirmation is replaced by the real call.
        assert [call.dump() for call in invocation.tool_calls] == [
            {"name": "execute", "input": {"code": "1+1"}, "result": 2}
        ]

    def test_fails_on_tool_confirmation_when_not_allowed(self) -> None:
        service = FakeService(
            chat=[
                chat_response(
                    actions=[
                        {"type": "toolConfirmation", "actionId": "a1", "toolConfirmation": {"toolName": "execute"}}
                    ]
                )
            ]
        )

        with pytest.raises(AgentEvalError, match=r"Agent requested tool confirmation \(execute\)"):
            invoke_agent_case(service, "my_agent", "Compute", allow_tool_confirmation=False)


EXPECTED_JUDGE_SYSTEM_MESSAGE = (
    "Respond with a single valid JSON object only. No markdown code fences or extra text.\n"
    'The JSON must conform to this schema: {"$schema":"http://json-schema.org/draft-07/schema#","type":"object",'
    '"properties":{"explanation":{"type":"string"},"label":{"type":"string","enum":["correct","incorrect"]}},'
    '"required":["explanation","label"],"additionalProperties":false}'
)


def make_case() -> EvalCase:
    """A two-turn case where the second turn has a correctness and a toolSelection scorer."""
    with TemporaryDirectory() as directory:
        write_files(
            Path(directory),
            {
                "eval/eval.yaml": "cases:\n  - id: c\n    turns:\n      - input: First\n      - input: Second\n"
                "        scorers:\n          - type: correctness\n            criteria: Mentions pumps\n"
                "          - type: toolSelection\n"
            },
        )
        return load_eval_cases(Path(directory))[0]


class TestJudge:
    def test_correctness_request_matches_cli(self) -> None:
        service = FakeService(completions=['{"explanation": "Mentions pumps", "label": "correct"}'])
        case = make_case()
        run = CaseRun("c", [TurnRun("First", "Hello", [], 1), TurnRun("Second", "Pumps!", [], 1)])
        judge = CdfJudge("azure/gpt-5", service)
        scorers = case.turns[1].scorers
        assert scorers is not None

        score = evaluate_scorer(case, 1, scorers[0], run, judge)

        assert score is not None and (score.passed, score.score, score.label) == (True, 1, "correct")
        request = service.completion_requests[0].dump()
        assert request["model"] == "azure/gpt-5"
        assert request["stream"] is False
        assert "temperature" not in request and "maxTokens" not in request
        system, user = request["messages"]
        assert system == {"role": "system", "content": EXPECTED_JUDGE_SYSTEM_MESSAGE}
        assert (
            "<input>\nConversation so far:\nUser: First\nAssistant: Hello\nUser question:\nSecond\nCriteria:\n"
            "Mentions pumps\n</input>" in user["content"]
        )
        assert "<output>\nPumps!\n</output>" in user["content"]

    def test_tool_scorer_without_tools_is_skipped(self) -> None:
        warnings: list[str] = []
        service = FakeService(completions=['{"explanation": "ok", "label": "correct"}'])
        case = make_case()
        run = CaseRun("c", [TurnRun("First", "Hello", [], 1), TurnRun("Second", "Pumps!", [], 1)])
        score_case = make_case_scorer(CdfJudge("m", service), warn=lambda message, hint: warnings.append(message))

        scores = score_case(case, run)

        assert [score.scorer_type for score in scores] == ["correctness"]
        assert warnings == ['Warning: Skipping toolSelection scorer on case "c": no tools available.']

    @pytest.mark.parametrize(
        "scorer_type, result, expected",
        [
            ("faithfulness", {"label": "faithful", "score": 0}, True),
            ("faithfulness", {"label": "unfaithful", "score": 1}, False),
            ("correctness", {"label": "other", "score": 0.5}, True),
            ("correctness", {"label": "other"}, False),
        ],
    )
    def test_scorer_passed(self, scorer_type: str, result: dict, expected: bool) -> None:
        assert scorer_passed(scorer_type, result) is expected

    def test_judge_model_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
        assert resolve_judge_model_id(None, None) == "azure/gpt-5"
        assert resolve_judge_model_id(None, "agent-model") == "agent-model"
        monkeypatch.setenv("EVAL_JUDGE_MODEL", "env-model")
        assert resolve_judge_model_id(None, "agent-model") == "env-model"
        assert resolve_judge_model_id(" flag-model ", "agent-model") == "flag-model"


AGENT_YAML = """externalId: my_agent
name: My agent
model: azure/gpt-4.1
tools:
  - type: askDocument
    name: ask_document
    description: Answers questions about documents
"""

DEPLOYED_AGENT = {
    "externalId": "my_agent",
    "name": "My agent",
    "model": "azure/gpt-4.1",
    "tools": [{"type": "askDocument", "name": "ask_document", "description": "Answers questions about documents"}],
    "createdTime": 1,
    "lastUpdatedTime": 2,
    "ownerId": "me",
    "runtimeVersion": "1.0.0",
}


@pytest.mark.usefixtures("disable_gzip")
class TestAgentEvalCommand:
    def test_runs_suite_and_writes_report(
        self,
        tmp_path: Path,
        toolkit_config: ToolkitClientConfig,
        respx_mock: respx.MockRouter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for name in ("EVAL_JUDGE_MODEL", "AGENT_CHAT_ROUTE", "AGENT_CHAT_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        agent_dir = write_files(
            tmp_path / "my_agent",
            {
                "my_agent.agent.yaml": AGENT_YAML,
                "eval/eval.yaml": """cases:
  - id: greeting
    tags: [ci]
    turns:
      - input: Hi
        scorers:
          - type: correctness
            criteria: Greets the user
  - id: refuses
    expectPass: false
    turns:
      - input: Delete everything
        scorers:
          - type: correctness
            criteria: Deletes all data
""",
            },
        )
        respx_mock.post(toolkit_config.create_api_url("/ai/agents/byids")).mock(
            return_value=httpx2.Response(200, json={"items": [DEPLOYED_AGENT]})
        )
        chat = respx_mock.post(toolkit_config.create_api_url("/ai/agents/chat")).mock(
            side_effect=[
                httpx2.Response(200, json=chat_response("Hello! I can help with documents.")),
                httpx2.Response(200, json=chat_response("I cannot do that.")),
            ]
        )
        completions = respx_mock.post(toolkit_config.create_api_url("/ai/chat/completions")).mock(
            side_effect=[
                httpx2.Response(
                    200, json={"choices": [{"message": {"content": '{"explanation": "Greets", "label": "correct"}'}}]}
                ),
                httpx2.Response(
                    200,
                    json={"choices": [{"message": {"content": '{"explanation": "Refused", "label": "incorrect"}'}}]},
                ),
            ]
        )
        stdout, stderr = StringIO(), StringIO()
        client = ToolkitClient(toolkit_config, console=Console(file=stdout, width=200))

        result = AgentEvalCommand(skip_tracking=True).eval(
            client, directory=agent_dir, report_dir=tmp_path / "report", stderr=Console(file=stderr, width=200)
        )

        assert result.has_failures is False
        assert chat.call_count == 2
        assert json.loads(completions.calls[0].request.content)["model"] == "azure/gpt-4.1"
        output = stdout.getvalue()
        assert "Judge model: azure/gpt-4.1" in output
        assert "✓ greeting" in output and "✓ refuses (expected fail)" in output
        assert "2/2 cases met expectations" in output
        # The deployed agent has a runtimeVersion that the local file does not set.
        assert "Warning: local *.agent.yaml differs from deployed config." in stderr.getvalue()
        assert "Fields: runtimeVersion" in stderr.getvalue()

        report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
        assert report["version"] == 1
        assert report["meta"]["agentExternalId"] == "my_agent"
        assert report["meta"]["route"] == "public"
        assert {key: report["summary"][key] for key in ("totalCases", "passedCases", "metExpectations")} == {
            "totalCases": 2,
            "passedCases": 1,
            "metExpectations": 2,
        }
        first = report["cases"][0]
        assert first["id"] == "greeting" and first["tags"] == ["ci"] and first["verdictPassed"] is True
        assert first["turns"] == [
            {"input": "Hi", "output": "Hello! I can help with documents.", "toolCalls": [], "apiRoundTrips": 1}
        ]
        assert first["scores"] == [
            {
                "turnIndex": 0,
                "scorerType": "correctness",
                "score": 1,
                "label": "correct",
                "explanation": "Greets",
                "passed": True,
            }
        ]
        markdown = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
        assert markdown.startswith("# Agent eval report\n")
        assert "## PASS: `refuses` (expected fail)" in markdown

    def test_upsert_requires_local_agent(self, tmp_path: Path, toolkit_config: ToolkitClientConfig) -> None:
        with pytest.raises(AgentEvalError, match="--upsert cannot be used with --external-id"):
            AgentEvalCommand(skip_tracking=True).eval(
                ToolkitClient(toolkit_config), directory=tmp_path, external_id="x", upsert=True
            )
