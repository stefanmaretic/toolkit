"""`cdf agents eval`: run an eval suite against an Atlas AI agent, the same way as `cognite agents eval`."""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console
from rich.markup import escape

from cognite_toolkit._cdf_tk.client import ToolkitClient
from cognite_toolkit._cdf_tk.client.http_client import ToolkitAPIError
from cognite_toolkit._cdf_tk.client.identifiers import ExternalId
from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import ChatContextInformation
from cognite_toolkit._cdf_tk.commands._base import ToolkitCommand

from ._agent_file import AGENT_FILE_GLOB, LocalAgent, detect_agent_drift, format_agent_drift_warning, read_local_agent
from ._errors import AgentEvalError
from ._judge import CdfJudge, judge_model_source, make_case_scorer, resolve_judge_model_id
from ._report import write_report
from ._runner import (
    TOOL_TEXT_LIMIT,
    UNSET,
    CaseResult,
    ChatRoute,
    ConsoleEvalLogger,
    EvalLogger,
    EvalService,
    EvalSummary,
    TeeEvalLogger,
    TraceCapture,
    case_verdict_passed,
    count_verdict_passes,
    format_tool_invocation,
    format_tool_result_preview,
    is_unexpected_pass,
    run_eval,
)
from ._suite import EvalCase, filter_eval_cases, load_eval_cases

SYSTEM_AGENT_EXTERNAL_ID = "cdf_industrial_v4"
BUILTIN_AGENT_PREFIX = "cdf_"
PUBLISHED_LABEL = "published"
AGENT_CHAT_ROUTE_ENV = "AGENT_CHAT_ROUTE"
AGENT_CHAT_BASE_URL_ENV = "AGENT_CHAT_BASE_URL"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1"})


def is_builtin_agent(external_id: str | None) -> bool:
    return external_id is not None and external_id.startswith(BUILTIN_AGENT_PREFIX)


def resolve_agent_chat_route_from_env() -> ChatRoute:
    value = (os.environ.get(AGENT_CHAT_ROUTE_ENV) or "").strip().lower()
    return "internal" if value in ("internal", "fusion") else "public"


def resolve_agent_chat_base_url() -> str | None:
    """The `AGENT_CHAT_BASE_URL` origin, for running against a local agent runtime. Loopback hosts only."""
    value = (os.environ.get(AGENT_CHAT_BASE_URL_ENV) or "").strip()
    if not value:
        return None
    default_hint = (
        "Pass an origin only, e.g. http://127.0.0.1:5000. Do not set COGNITE_BASE_URL to localhost — use "
        f"{AGENT_CHAT_BASE_URL_ENV} instead."
    )

    def invalid(reason: str, hint: str = default_hint) -> AgentEvalError:
        return AgentEvalError(f'Invalid agent chat base URL "{value}": {reason}.', hint=hint)

    try:
        url = urlsplit(value)
        hostname = url.hostname
    except ValueError:
        raise invalid("must be a valid URL") from None
    if not url.scheme or not url.netloc:
        raise invalid("must be a valid URL")
    if url.scheme not in ("http", "https"):
        raise invalid("must use http or https")
    if url.path not in ("", "/"):
        raise invalid("must be an origin only (e.g. http://127.0.0.1:5000), not a full API path")
    if url.query or url.fragment:
        raise invalid("must not include query or fragment")
    if hostname not in _LOOPBACK_HOSTS:
        raise invalid(
            "must target a loopback host (localhost or 127.0.0.1)",
            "Use a loopback origin for local cog-ai. Override chat POSTs include the CLI CDF access token; they "
            "are only sent to loopback origins.",
        )
    return f"{url.scheme}://{url.netloc}"


def format_session_context_summary(context: ChatContextInformation | None) -> str:
    if context is None:
        return "(none)"
    parts: list[str] = []
    model_count, space_count = len(context.data_models or []), len(context.instance_spaces or [])
    if model_count or space_count:
        parts.append(f"{model_count} data model(s), {space_count} space(s)")
    if context.app_context:
        parts.append("app context")
    if context.skills:
        parts.append(f"{len(context.skills)} skill(s)")
    if context.subagents:
        parts.append(f"{len(context.subagents)} subagent(s)")
    return ", ".join(parts) if parts else "(none)"


def summarize_session_context(cases: Sequence[EvalCase]) -> str | None:
    with_context = [case for case in cases if case.context_information is not None]
    if not with_context:
        return None
    first = with_context[0].context_information
    if len(with_context) == len(cases) and all(case.context_information == first for case in with_context):
        return format_session_context_summary(first)
    return f"varies by case ({len(with_context)}/{len(cases)})"


def _indent(text: str, level: int = 1) -> str:
    return "  " * level + text


@dataclass
class AgentEvalResult:
    summary: EvalSummary
    has_failures: bool
    report_paths: tuple[Path, Path] | None = None


class AgentEvalCommand(ToolkitCommand):
    """Runs eval suites (`eval/eval.yaml`) against Atlas AI agents."""

    def eval(
        self,
        client: ToolkitClient,
        directory: Path | None = None,
        case_filter: Sequence[str] | None = None,
        tag_filter: Sequence[str] | None = None,
        judge_model: str | None = None,
        external_id: str | None = None,
        system_agent: bool = False,
        upsert: bool = False,
        fail_fast: bool = False,
        verbose: bool = False,
        fail_on_tool_confirmation: bool = False,
        report_dir: Path | None = None,
        stderr: Console | None = None,
    ) -> AgentEvalResult:
        """Run the eval cases of an agent project directory.

        Args:
            client: Toolkit client used to call CDF.
            directory: Agent project directory, with `<name>.agent.yaml` and `eval/eval.yaml`. Defaults to the
                current working directory.
            case_filter: Only run cases with these ids.
            tag_filter: Only run cases with at least one of these tags.
            judge_model: Language model for the judge.
            external_id: Evaluate this deployed agent instead of the agent in `<name>.agent.yaml`.
            system_agent: Evaluate the Cognite system agent.
            upsert: Push the local agent definition before running the cases.
            fail_fast: Stop after the first case that does not meet expectations.
            verbose: Log request/response traces to stderr.
            fail_on_tool_confirmation: Fail a case instead of approving tool confirmations.
            report_dir: Write report.json and report.md to this directory.
            stderr: Console for warnings and verbose traces. Defaults to stderr.

        Returns:
            The eval summary. `has_failures` is True if any case did not meet expectations.
        """
        out = client.console
        err = stderr or Console(stderr=True, soft_wrap=True)

        def warn(message: str) -> None:
            err.print(f"[yellow]{escape(message)}[/]")

        directory = (directory or Path.cwd()).resolve()
        if external_id is not None and system_agent:
            raise AgentEvalError("--external-id and --system-agent cannot be used together.")
        if upsert and (external_id is not None or system_agent):
            raise AgentEvalError("--upsert cannot be used with --external-id or --system-agent.")

        agent_external_id, local_agent = self._resolve_agent_identity(directory, external_id, system_agent)
        if upsert and local_agent is None:
            raise AgentEvalError(
                f"--upsert requires a local {AGENT_FILE_GLOB} to push.",
                hint=f"Remove --upsert, or add an agent definition ({AGENT_FILE_GLOB}) to this directory.",
            )

        all_cases = load_eval_cases(directory)
        cases = filter_eval_cases(all_cases, case_filter, tag_filter)
        self._assert_runtime_subagents_allowed(agent_external_id, cases)
        sources = list(dict.fromkeys(case.source for case in all_cases))
        multi_source = len(sources) > 1
        cases = self._filter_unpublished_subagents(client, cases, warn)

        if local_agent is not None:
            self._ensure_agent_config(client, local_agent, upsert, verbose, warn)

        chat_base_url = resolve_agent_chat_base_url()
        route: ChatRoute = (
            "internal"
            if chat_base_url is not None or is_builtin_agent(agent_external_id)
            else resolve_agent_chat_route_from_env()
        )
        service = EvalService(client, route=route, chat_base_url=chat_base_url)
        trace = TraceCapture() if report_dir is not None else None
        console_logger = ConsoleEvalLogger(Console(stderr=True, markup=False, highlight=False)) if verbose else None
        logger: EvalLogger | None
        if trace is not None and console_logger is not None:
            logger = TeeEvalLogger(console_logger, trace)
        else:
            logger = trace or console_logger
        agent_model = local_agent.request.model if local_agent is not None else None
        judge_model_id = resolve_judge_model_id(judge_model, agent_model)

        if multi_source:
            out.print(f"Matched {len(cases)} of {len(all_cases)} case(s) across {len(sources)} file(s)")
        session_summary = summarize_session_context(cases)
        if session_summary is not None:
            out.print(f"Session context: {escape(session_summary)}")
        out.print(f"Judge model: [cyan]{escape(judge_model_id)}[/]")
        if logger:
            logger.log(f"Agent: {agent_external_id}")
            for case in cases:
                logger.log(f"  {case.id}: {format_session_context_summary(case.context_information)}")
            for source in sources:
                logger.log(f"  {source} ({sum(1 for case in all_cases if case.source == source)} case(s))")
            logger.log(f"Loaded {len(cases)} case(s): {', '.join(case.id for case in cases)}")
            logger.log(
                f"judge model: {judge_model_id} (source: {judge_model_source(judge_model, agent_model, AGENT_FILE_GLOB)})"
            )
            logger.log(f"agent chat route: {route}")
            logger.log(f"agent chat URL: {service.agent_chat_url}")
            logger.log(f"chat completions URL: {service.chat_completions_url}")

        judge = CdfJudge(judge_model_id, service, logger)
        score_case = make_case_scorer(
            judge,
            agent_tools=local_agent.eval_tools() if local_agent is not None else None,
            warn=lambda message, hint: warn(f"{message}\n  hint: {hint}" if hint else message),
            agent_file_label=AGENT_FILE_GLOB,
        )
        summary = run_eval(
            cases,
            service,
            agent_external_id,
            score_case,
            allow_tool_confirmation=not fail_on_tool_confirmation,
            fail_fast=fail_fast,
            logger=logger,
        )

        for result in summary.results:
            self._print_case_result(result, out.print, multi_source, verbose)
            if is_unexpected_pass(result):
                warn(f'Warning: case "{result.eval_case.id}" expected scorers to fail but they passed.')

        has_failures = any(not case_verdict_passed(result) for result in summary.results)
        met = f"{count_verdict_passes(summary.results)}/{len(summary.results)} cases met expectations"
        out.print(f"\n[bold {'red' if has_failures else 'green'}]{met}[/]")
        if summary.skipped_cases > 0:
            out.print(f"[yellow]Stopped early (--fail-fast): {summary.skipped_cases} case(s) not run.[/]")

        report_paths: tuple[Path, Path] | None = None
        if report_dir is not None and trace is not None:
            try:
                report_paths = write_report(
                    report_dir.resolve(),
                    summary,
                    trace.events,
                    {"agentExternalId": agent_external_id, "judgeModelId": judge_model_id, "route": route},
                )
            except OSError as error:
                warn(f"Warning: failed to write eval report: {error}")
            else:
                for path in report_paths:
                    out.print(f"Wrote eval report: {escape(str(path))}")
        return AgentEvalResult(summary=summary, has_failures=has_failures, report_paths=report_paths)

    @staticmethod
    def _resolve_agent_identity(
        directory: Path, external_id: str | None, system_agent: bool
    ) -> tuple[str, LocalAgent | None]:
        if external_id is not None:
            if not external_id.strip():
                raise AgentEvalError(
                    "--external-id cannot be empty.",
                    hint=f"Provide a valid agent external ID, or omit the flag to read from {AGENT_FILE_GLOB}.",
                )
            return external_id.strip(), None
        if system_agent:
            return SYSTEM_AGENT_EXTERNAL_ID, None
        local_agent = read_local_agent(directory)
        if not local_agent.external_id:
            raise AgentEvalError(
                "No agent specified.",
                hint=f"Provide --external-id, --system-agent, or run from a directory with {AGENT_FILE_GLOB}.",
            )
        return local_agent.external_id, local_agent

    @staticmethod
    def _assert_runtime_subagents_allowed(agent_external_id: str, cases: Sequence[EvalCase]) -> None:
        if is_builtin_agent(agent_external_id):
            return
        with_subagents = [case.id for case in cases if case.context_information and case.context_information.subagents]
        if with_subagents:
            raise AgentEvalError(
                "Runtime subagents are only supported when evaluating a Cognite builtin agent (external ID "
                "starting with `cdf_`).",
                hint="Remove runtime subagents (additionalSubagents or includeExtension) from: "
                f"{', '.join(with_subagents)}. Or evaluate a Cognite builtin agent via --system-agent or "
                "--external-id with a cdf_ ID.",
            )

    @staticmethod
    def _filter_unpublished_subagents(
        client: ToolkitClient, cases: list[EvalCase], warn: Callable[[str], None]
    ) -> list[EvalCase]:
        """Drop subagents from a published extension that do not exist or are not published, as Atlas does."""
        wanted = list(dict.fromkeys(name for case in cases for name in case.published_subagents or []))
        if not wanted:
            return cases
        agents = {agent.external_id: agent for agent in client.agents.list(limit=None)}
        missing = [name for name in wanted if name not in agents]
        unpublished = [name for name in wanted if name in agents and PUBLISHED_LABEL not in (agents[name].labels or [])]
        for name in unpublished:
            warn(f'Warning: subagent "{name}" in the published includeExtension is not published; ignoring it.')
        for name in missing:
            warn(f'Warning: subagent "{name}" in the published includeExtension does not exist; ignoring it.')
        ignored = {*missing, *unpublished}
        if not ignored:
            return cases
        output: list[EvalCase] = []
        for case in cases:
            context = case.context_information
            if not case.published_subagents or context is None or not context.subagents:
                output.append(case)
                continue
            kept = [name for name in context.subagents if name not in ignored]
            values = context.model_dump(exclude_unset=True)
            if kept:
                values["subagents"] = kept
            else:
                values.pop("subagents", None)
            output.append(
                EvalCase(
                    id=case.id,
                    turns=case.turns,
                    source=case.source,
                    expect_pass=case.expect_pass,
                    tags=case.tags,
                    context_information=ChatContextInformation.model_validate(values),
                    tools=case.tools,
                    published_subagents=case.published_subagents,
                )
            )
        return output

    @staticmethod
    def _ensure_agent_config(
        client: ToolkitClient, local_agent: LocalAgent, upsert: bool, verbose: bool, warn: Callable[[str], None]
    ) -> None:
        if upsert:
            client.agents.create([local_agent.request])
            client.console.print(
                f'[green]Pushed agent "{escape(local_agent.external_id)}" to project '
                f'"{escape(client.config.project)}"[/]'
            )
            return
        try:
            deployed = client.agents.retrieve(
                [ExternalId(external_id=local_agent.external_id)], ignore_unknown_ids=True
            )
        except ToolkitAPIError as error:
            warn(f'Warning: failed to fetch deployed agent "{local_agent.external_id}": {error.message}')
            return
        if not deployed:
            warn(
                f'Warning: agent "{local_agent.external_id}" is not deployed in CDF. Run with --upsert to push local '
                "config before evaluating."
            )
            return
        drift = detect_agent_drift(local_agent.request, deployed[0])
        if drift:
            warn(format_agent_drift_warning(drift, verbose=verbose))

    @staticmethod
    def _print_case_result(result: CaseResult, write: Callable[[str], None], multi_source: bool, verbose: bool) -> None:
        case = result.eval_case
        mark = "[green]✓[/]" if case_verdict_passed(result) else "[red]✗[/]"
        name = f"{case.id} ({case.source})" if multi_source and case.source else case.id
        suffix = " [dim](expected fail)[/]" if not case.expect_pass else ""
        write(f"{mark} {escape(name)}{suffix}")
        for label, error in (("invoke", result.invoke_error), ("score", result.score_error)):
            if error is None:
                continue
            write(_indent(f"[red]{label}: fail[/]"))
            write(_indent(escape(error.message if isinstance(error, AgentEvalError) else str(error)), 2))
            if isinstance(error, AgentEvalError) and error.hint:
                write(_indent(f"[yellow]💡 {escape(error.hint)}[/]", 2))
            if label == "invoke":
                return

        turn_count = len(result.run.turns) if result.run is not None else 0
        indices = sorted({*range(turn_count), *(score.turn_index for score in result.scores)})
        multi_turn = len(indices) > 1
        for index in indices:
            level = 2 if multi_turn else 1
            if multi_turn:
                write(_indent(f"[dim]turn {index + 1}[/]"))
            if result.run is not None and index < turn_count and result.run.turns[index].tool_calls:
                write(_indent("[dim]toolsCalled:[/]", level))
                for call in result.run.turns[index].tool_calls:
                    text = escape(format_tool_invocation(call, TOOL_TEXT_LIMIT))
                    if verbose and call.result is not UNSET and call.result is not None:
                        text += f" [dim]→ {escape(format_tool_result_preview(call.result))}[/]"
                    write(_indent(text, level + 1))
            for score in (score for score in result.scores if score.turn_index == index):
                verdict = "[green]pass[/]" if score.passed else "[red]fail[/]"
                write(_indent(f"{escape(score.scorer_type)}: {verdict}", level))
                if score.explanation:
                    write(_indent(f"[dim]{escape(score.explanation)}[/]", level + 1))
