"""The eval report (`report.json` and `report.md`), in the same format as `cognite agents eval --report-dir`."""

import json
import re
from pathlib import Path
from typing import Any

from ._errors import AgentEvalError
from ._runner import (
    TOOL_TEXT_LIMIT,
    UNSET,
    CaseResult,
    EvalSummary,
    TurnRun,
    case_verdict_passed,
    count_unexpected_passes,
    count_verdict_failures,
    count_verdict_passes,
    format_duration_ms,
    format_tool_invocation,
    format_tool_result_preview,
)

REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"
DEFAULT_REPORT_DIR = "eval-report"
REPORT_VERSION = 1
_MAX_TRACES_IN_MARKDOWN = 40


def serialize_error(error: BaseException) -> dict[str, Any]:
    output: dict[str, Any] = {
        "name": type(error).__name__,
        "message": error.message if isinstance(error, AgentEvalError) else str(error),
    }
    if isinstance(error, AgentEvalError):
        if error.hint:
            output["hint"] = error.hint
        if error.http_status is not None:
            http: dict[str, Any] = {"status": error.http_status}
            if error.request_url is not None:
                http["url"] = error.request_url
            if error.response_body is not None:
                http["responseBody"] = error.response_body
            output["http"] = http
    if isinstance(error.__cause__, Exception):
        output["cause"] = serialize_error(error.__cause__)
    return output


def _serialize_case(result: CaseResult) -> dict[str, Any]:
    case = result.eval_case
    output: dict[str, Any] = {"id": case.id, "source": case.source}
    if case.tags:
        output["tags"] = case.tags
    output["expectPass"] = case.expect_pass
    output["scorersPassed"] = result.passed
    output["verdictPassed"] = case_verdict_passed(result)
    if case.context_information is not None:
        output["contextInformation"] = case.context_information.dump()
    if result.run is not None:
        output["turns"] = [
            {
                "input": turn.input,
                "output": turn.output,
                "toolCalls": [call.dump() for call in turn.tool_calls],
                "apiRoundTrips": turn.api_round_trips,
            }
            for turn in result.run.turns
        ]
    scores: list[dict[str, Any]] = []
    for score in result.scores:
        item: dict[str, Any] = {"turnIndex": score.turn_index, "scorerType": score.scorer_type, "score": score.score}
        if score.label is not None:
            item["label"] = score.label
        if score.explanation is not None:
            item["explanation"] = score.explanation
        item["passed"] = score.passed
        scores.append(item)
    output["scores"] = scores
    if result.invoke_error is not None:
        output["invokeError"] = serialize_error(result.invoke_error)
    if result.score_error is not None:
        output["scoreError"] = serialize_error(result.score_error)
    output["chatDurationMs"] = result.chat_duration_ms
    if result.judge_duration_ms is not None:
        output["judgeDurationMs"] = result.judge_duration_ms
    output["durationMs"] = result.duration_ms
    return output


def build_report_document(summary: EvalSummary, traces: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    results = summary.results
    return {
        "version": REPORT_VERSION,
        "meta": meta,
        "summary": {
            "totalCases": summary.total_cases,
            "passedCases": summary.passed_cases,
            "failedCases": summary.failed_cases,
            "skippedCases": summary.skipped_cases,
            "metExpectations": count_verdict_passes(results),
            "verdictFailures": count_verdict_failures(results),
            "unexpectedPassCount": count_unexpected_passes(results),
            "chatDurationMs": sum(result.chat_duration_ms for result in results),
            "judgeDurationMs": sum(result.judge_duration_ms or 0 for result in results),
            "durationMs": sum(result.duration_ms for result in results),
        },
        "cases": [_serialize_case(result) for result in results],
        "traces": traces,
    }


def write_report(
    report_dir: Path, summary: EvalSummary, traces: list[dict[str, Any]], meta: dict[str, Any]
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    document = build_report_document(summary, traces, {**meta, "reportDir": str(report_dir)})
    json_path = report_dir / REPORT_JSON
    markdown_path = report_dir / REPORT_MARKDOWN
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(format_report_markdown(document, summary), encoding="utf-8")
    return json_path, markdown_path


def _fence(text: str) -> str:
    return "```\n" + text.replace("```", "``\\`") + "\n```"


def _inline_code(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") or text.startswith(" ") or text.endswith(" ") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def _lines(*items: str | None) -> str:
    return "\n".join(item for item in items if item)


def _trim_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.lstrip("\n")).rstrip() + "\n"


def _format_suite_duration(summary: dict[str, Any]) -> str:
    return (
        f"{format_duration_ms(summary['durationMs'])} (chat {format_duration_ms(summary['chatDurationMs'])}, "
        f"judge {format_duration_ms(summary['judgeDurationMs'])})"
    )


def _format_case_duration(case: dict[str, Any]) -> str:
    judge = "judge skipped" if "judgeDurationMs" not in case else f"judge {format_duration_ms(case['judgeDurationMs'])}"
    return f"{format_duration_ms(case['durationMs'])} (chat {format_duration_ms(case['chatDurationMs'])}, {judge})"


def _format_http_error(http: dict[str, Any]) -> str:
    body = None
    if "responseBody" in http:
        raw = http["responseBody"]
        body = "- Response body:\n\n" + _fence(
            raw if isinstance(raw, str) else json.dumps(raw, indent=2, ensure_ascii=False)
        )
    return _lines(f"- HTTP status: `{http['status']}`", http.get("url") and f"- URL: `{http['url']}`", body)


def _format_cause(cause: dict[str, Any]) -> str:
    http = cause.get("http")
    return _lines(
        f"- Cause: {cause['name']}: {cause['message']}",
        http and f"  - HTTP status: `{http['status']}`",
        http and http.get("url") and f"  - URL: `{http['url']}`",
    )


def _format_error(title: str, error: dict[str, Any]) -> str:
    body = _lines(
        f"- Message: {error['message']}",
        f"- Hint: {error['hint']}" if error.get("hint") else None,
        f"- Help: {error['helpUrl']}" if error.get("helpUrl") else None,
        _format_http_error(error["http"]) if "http" in error else None,
        _format_cause(error["cause"]) if "cause" in error else None,
    )
    return f"\n### {title}\n\n{body}\n"


def _format_scores(scores: list[dict[str, Any]], show_turn_index: bool = False) -> str:
    lines: list[str] = []
    for score in scores:
        turn = f" (turn {score['turnIndex'] + 1})" if show_turn_index else ""
        details = ", ".join(
            part for part in (score.get("label") and f"label={score['label']}", f"score={score['score']}") if part
        )
        explanation = f"\n  - {score['explanation']}" if score.get("explanation") else ""
        lines.append(
            f"- `{score['scorerType']}`{turn}: **{'pass' if score['passed'] else 'fail'}** ({details}){explanation}"
        )
    return "\n**Scorers**\n\n" + "\n".join(lines) + "\n"


def _format_tool_calls(turn: TurnRun) -> str:
    if not turn.tool_calls:
        return ""
    lines = []
    for call in turn.tool_calls:
        text = format_tool_invocation(call, TOOL_TEXT_LIMIT)
        if call.result is not UNSET:
            text += f" → {format_tool_result_preview(call.result, TOOL_TEXT_LIMIT)}"
        lines.append(f"- {_inline_code(text)}")
    return "\n**Tool calls**\n\n" + "\n".join(lines) + "\n"


def _format_turn(turn: TurnRun, turn_index: int, case: dict[str, Any], turn_count: int) -> str:
    heading = f"### Turn {turn_index + 1}" if turn_count > 1 else "### Turn"
    scores = [score for score in case["scores"] if score["turnIndex"] == turn_index]
    return (
        f"\n{heading}\n\n**Prompt**\n\n{_fence(turn.input)}\n\n**Response**\n\n{_fence(turn.output or '(empty)')}\n\n"
        f"{_format_tool_calls(turn)}{_format_scores(scores) if scores else ''}\n"
    )


def _format_case(case: dict[str, Any], result: CaseResult) -> str:
    verdict = "PASS" if case["verdictPassed"] else "FAIL"
    expected_fail = " (expected fail)" if case["expectPass"] is False else ""
    tags = f"- Tags: {', '.join(f'`{tag}`' for tag in case['tags'])}\n" if case.get("tags") else ""
    errors = ""
    if "invokeError" in case:
        errors += _format_error("Invoke error", case["invokeError"])
    if "scoreError" in case:
        errors += _format_error("Score error", case["scoreError"])
    if result.run is not None:
        turns = result.run.turns
        body = "\n".join(_format_turn(turn, index, case, len(turns)) for index, turn in enumerate(turns))
    elif case["scores"]:
        body = _format_scores(case["scores"], show_turn_index=True)
    else:
        body = ""
    return (
        f"\n## {verdict}: `{case['id']}`{expected_fail}\n\n"
        f"- Source: `{case['source']}`\n"
        f"- Verdict: {'met expectations' if case['verdictPassed'] else 'did not meet expectations'}\n"
        f"- Scorers: {'all passed' if case['scorersPassed'] else 'failed or missing'}\n"
        f"- Duration: {_format_case_duration(case)}\n"
        f"{tags}\n{errors}{body}\n"
    )


def _format_traces(traces: list[dict[str, Any]]) -> str:
    if not traces:
        return ""
    shown = traces[:_MAX_TRACES_IN_MARKDOWN]
    lines = "\n".join(
        f"- {event['message']}" if event["type"] == "log" else f"- JSON: {event['label']}" for event in shown
    )
    more = f"\n- … {len(traces) - len(shown)} more event(s) in `{REPORT_JSON}`" if len(traces) > len(shown) else ""
    return (
        f"\n## Debug traces (summary)\n\nCaptured {len(traces)} verbose-equivalent event(s). See `{REPORT_JSON}` for "
        "full payloads (agent chat request/response, judge inputs, chat/completions I/O).\n\n"
        f"{lines}{more}\n"
    )


def format_report_markdown(document: dict[str, Any], summary: EvalSummary) -> str:
    meta, totals, cases = document["meta"], document["summary"], document["cases"]
    skipped = f" ({totals['skippedCases']} skipped)" if totals["skippedCases"] > 0 else ""
    case_sections = "\n".join(_format_case(case, result) for case, result in zip(cases, summary.results))
    return _trim_blank_lines(
        f"\n# Agent eval report\n\n"
        f"- Agent: `{meta['agentExternalId']}`\n"
        f"- Judge model: `{meta['judgeModelId']}`\n"
        f"- Chat route: `{meta['route']}`\n"
        f"- Summary: **{totals['metExpectations']}/{len(cases)}** cases met expectations{skipped}\n"
        f"- Duration: {_format_suite_duration(totals)}\n"
        f"- Full verbose-equivalent traces: `{REPORT_JSON}` ({len(document['traces'])} event(s))\n\n"
        f"{case_sections}\n{_format_traces(document['traces'])}\n"
    )
