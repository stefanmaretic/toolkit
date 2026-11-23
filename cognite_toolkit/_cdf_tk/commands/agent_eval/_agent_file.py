"""Reading the local agent definition (`<name>.agent.yaml`) that an eval runs against."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cognite_toolkit._cdf_tk.client.identifiers import ExternalId
from cognite_toolkit._cdf_tk.client.resource_classes.agent import AgentRequest, AgentResponse
from cognite_toolkit._cdf_tk.resource_ios import AgentIO
from cognite_toolkit._cdf_tk.resource_ios._base_ios import FailedReadExtra, SuccessExtra
from cognite_toolkit._cdf_tk.utils import sanitize_filename
from cognite_toolkit._cdf_tk.utils.file import read_yaml_content

from ._errors import AgentEvalError
from ._suite import EvalTool, format_validation_error

AGENT_FILE_GLOB = "*.agent.yaml"
_AGENT_FILE_SUFFIXES = (".agent.yaml", ".agent.yml")
_SERVER_FIELDS = ("createdTime", "lastUpdatedTime", "ownerId")
_SCALAR_FIELDS = ("name", "description", "instructions", "model", "runtimeVersion")
_ARRAY_FIELDS = ("labels", "tools", "skills", "subagents", "exampleQuestions")


@dataclass
class LocalAgent:
    path: Path
    request: AgentRequest

    @property
    def external_id(self) -> str:
        return self.request.external_id

    def eval_tools(self) -> list[EvalTool] | None:
        """The agent's tools, used by the tool scorers when the eval suite declares none."""
        if self.request.tools is None:
            return None
        return [EvalTool.model_construct(name=tool.name, description=tool.description) for tool in self.request.tools]


def find_agent_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.name.lower().endswith(_AGENT_FILE_SUFFIXES)),
        key=lambda path: path.name,
    )


def agent_file_name(directory: Path) -> str:
    return f"{sanitize_filename(directory.resolve().name)}.agent.yaml"


def resolve_agent_file(directory: Path) -> Path:
    candidates = find_agent_files(directory)
    if not candidates:
        raise AgentEvalError(
            f"No {AGENT_FILE_GLOB} found in:\n  {directory}",
            hint="Run this command from an agent project directory, or pass the directory as an argument.",
        )
    if len(candidates) == 1:
        return candidates[0]
    expected = agent_file_name(directory).lower()
    match = next((path for path in candidates if path.name.lower() == expected), None)
    if match is not None:
        return match
    listed = "\n".join(f"  - {path.name}" for path in candidates)
    raise AgentEvalError(
        f"Found multiple {AGENT_FILE_GLOB} files in:\n  {directory}\n{listed}",
        hint="Each agent needs its own directory. Move the extra definitions elsewhere, or name the one you want "
        f"after its folder ({agent_file_name(directory)}).",
    )


def read_local_agent(directory: Path) -> LocalAgent:
    """Read the agent definition in a directory.

    Toolkit agent files can reference an instructions file (`instructionsFile`) and tool files (`toolFiles`).
    These are inlined the same way as `cdf build` does. Toolkit variables are not resolved.
    """
    path = resolve_agent_file(directory)
    raw = path.read_text(encoding="utf-8-sig")
    if "{{" in raw:
        raise AgentEvalError(
            f"{path.name} contains Toolkit variables ({{{{ variable }}}}), which `cdf agents eval` does not resolve.",
            hint="Pass --external-id to evaluate the deployed agent instead.",
        )
    data = read_yaml_content(raw)
    if not isinstance(data, dict):
        raise AgentEvalError(f"Invalid {path}: expected a mapping with the agent definition.")
    data = _inline_extra_files(path, data)
    try:
        request = AgentRequest.model_validate(data)
    except ValidationError as error:
        raise AgentEvalError(f"Invalid {path}: {format_validation_error(error)}") from None
    return LocalAgent(path=path, request=request)


def _inline_extra_files(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    external_id = data.get("externalId")
    identifier = ExternalId(external_id=external_id if isinstance(external_id, str) and external_id else "unknown")
    extra_by_field: dict[str, list[SuccessExtra]] = {}
    for extra in AgentIO.get_extra_files(path, identifier, data):
        if isinstance(extra, FailedReadExtra):
            raise AgentEvalError(f"Invalid {path}: {extra.error}")
        if isinstance(extra, SuccessExtra) and extra.resource_field is not None:
            extra_by_field.setdefault(extra.resource_field, []).append(extra)
    output = dict(data)
    for field_name, extras in extra_by_field.items():
        if extras[0].is_list:
            output[field_name] = [
                extra.content_parsed or extra.content for extra in extras if extra.content_parsed or extra.content
            ]
        else:
            output[field_name] = extras[0].content_parsed or extras[0].content
        for remove_field in extras[0].remove_fields:
            output.pop(remove_field, None)
    return output


@dataclass
class AgentDrift:
    field: str
    local: str
    remote: str


def detect_agent_drift(local: AgentRequest, remote: AgentResponse) -> list[AgentDrift]:
    """The fields where the local agent definition differs from the deployed agent."""
    local_dump = _without_server_fields(local.dump())
    remote_dump = _without_server_fields(remote.dump())
    drift: list[AgentDrift] = []
    for name in _SCALAR_FIELDS:
        local_value, remote_value = local_dump.get(name) or "", remote_dump.get(name) or ""
        if local_value != remote_value:
            drift.append(AgentDrift(name, str(local_value), str(remote_value)))
    for name in _ARRAY_FIELDS:
        local_value = _stable_array(name, local_dump.get(name))
        remote_value = _stable_array(name, remote_dump.get(name))
        if local_value != remote_value:
            drift.append(AgentDrift(name, local_value, remote_value))
    return drift


def format_agent_drift_warning(drift: list[AgentDrift], verbose: bool = False) -> str:
    lines = [
        f"Warning: local {AGENT_FILE_GLOB} differs from deployed config.",
        f"  Fields: {', '.join(item.field for item in drift)}",
    ]
    if verbose:
        for item in drift:
            lines.append(f'   {item.field + ":":<18}"{item.remote}" → "{item.local}"')
    lines.append("  Run with --upsert to push local changes before evaluating.")
    return "\n".join(lines)


def _without_server_fields(dumped: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dumped.items() if key not in _SERVER_FIELDS}


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _array_item_sort_key(field_name: str, item: Any) -> str:
    if not isinstance(item, dict):
        return _stable(item)
    if field_name == "tools":
        return str(item.get("name", ""))
    if field_name == "subagents":
        return str(item.get("agentExternalId", ""))
    if field_name == "exampleQuestions":
        return str(item.get("question", ""))
    return _stable(item)


def _stable_array(field_name: str, value: Any) -> str:
    items = list(value or [])
    return _stable(sorted(items, key=lambda item: _array_item_sort_key(field_name, item)))
