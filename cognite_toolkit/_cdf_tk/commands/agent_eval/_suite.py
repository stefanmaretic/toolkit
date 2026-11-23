"""Eval suite files (`eval/eval.yaml` and included fragments) and how they are loaded into eval cases.

The schema and the loading rules match `cognite agents eval` in @cognite/cli, so that the same suite
can be run with either tool. The CLI's runtime (valibot) schema is the source of truth. The published
JSON schema at https://docs.cognite.com/assets/schemas/agent-eval-cases.schema.json does not list
`additionalInstructions`, `additionalSkills`, `additionalSubagents` and `includeExtension`.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator
from pydantic.alias_generators import to_camel
from yaml import YAMLError

from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import ChatContextInformation, ChatDataModel
from cognite_toolkit._cdf_tk.constants import AGENT_EVAL_DIRECTORY
from cognite_toolkit._cdf_tk.utils.file import read_yaml_content

from ._errors import AgentEvalError

EVAL_DIR = AGENT_EVAL_DIRECTORY
EVAL_CASES_FILE = "eval.yaml"
PROJECT_CONFIG_FILE = "agent-project-config.yaml"
MAX_ADDITIONAL_SUBAGENTS = 20
MAX_ADDITIONAL_INSTRUCTIONS = 8000
SCORER_TYPES = ("correctness", "faithfulness", "toolSelection", "toolInvocation")
_SKIP_WALK_DIRECTORIES = frozenset(
    {".git", "node_modules", "eval-report", "dist", "build", "coverage", "__pycache__", "venv", ".venv"}
)
_EVAL_SUITE_PATH = f"{EVAL_DIR}/{EVAL_CASES_FILE}"

T = TypeVar("T")


def _non_empty(message: str = "must not be empty") -> AfterValidator:
    def check(value: str) -> str:
        if not value:
            raise ValueError(message)
        return value

    return AfterValidator(check)


def _min_items(minimum: int, message: str) -> AfterValidator:
    def check(value: list[T]) -> list[T]:
        if len(value) < minimum:
            raise ValueError(message)
        return value

    return AfterValidator(check)


def _max_items(maximum: int, message: str) -> AfterValidator:
    def check(value: list[T]) -> list[T]:
        if len(value) > maximum:
            raise ValueError(message)
        return value

    return AfterValidator(check)


def _unique(message: str) -> AfterValidator:
    def check(value: list[T]) -> list[T]:
        if len(set(value)) != len(value):  # type: ignore[arg-type]
            raise ValueError(message)
        return value

    return AfterValidator(check)


def _max_length(maximum: int) -> AfterValidator:
    def check(value: str) -> str:
        if len(value) > maximum:
            raise ValueError(f"must be {maximum} characters or fewer")
        return value

    return AfterValidator(check)


NonEmptyStr = Annotated[str, _non_empty()]


class _EvalModel(BaseModel):
    # Unknown keys are dropped, as valibot's `object` does in the CLI.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class EvalTool(_EvalModel):
    """A tool the judge expects the agent to call (runtime name), used by the tool scorers."""

    name: NonEmptyStr
    description: NonEmptyStr


ToolList = Annotated[list[EvalTool], _min_items(1, "must contain at least one tool")]


class EvalDataModel(_EvalModel):
    space: NonEmptyStr
    external_id: NonEmptyStr
    version: NonEmptyStr


class CorrectnessScorer(_EvalModel):
    type: Literal["correctness"]
    criteria: NonEmptyStr


class FaithfulnessScorer(_EvalModel):
    type: Literal["faithfulness"]
    ground_truth: NonEmptyStr


class ToolSelectionScorer(_EvalModel):
    type: Literal["toolSelection"]
    tools: ToolList | None = None


class ToolInvocationScorer(_EvalModel):
    type: Literal["toolInvocation"]
    tools: ToolList | None = None


Scorer = Annotated[
    CorrectnessScorer | FaithfulnessScorer | ToolSelectionScorer | ToolInvocationScorer,
    Field(discriminator="type"),
]


class EvalTurn(_EvalModel):
    input: NonEmptyStr
    scorers: Annotated[list[Scorer], _min_items(1, "must contain at least one scorer")] | None = None


class IncludeExtension(BaseModel):
    """Points at an agent project config file whose draft or published system agent extension is used."""

    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, _non_empty("includeExtension path must not be empty")]
    type: Literal["draft", "published"]


class _LevelFields(_EvalModel):
    """Fields that can be set on the root file, an included file, and a case. The most specific level wins."""

    data_models: Annotated[list[EvalDataModel], _min_items(1, "must contain at least one data model")] | None = None
    instance_spaces: Annotated[list[NonEmptyStr], _min_items(1, "must contain at least one instance space")] | None = (
        None
    )
    app_context: NonEmptyStr | None = None
    additional_instructions: Annotated[str, _max_length(MAX_ADDITIONAL_INSTRUCTIONS)] | None = None
    additional_skills: Annotated[list[NonEmptyStr], _unique("skill IDs must be unique")] | None = None
    additional_subagents: (
        Annotated[
            list[NonEmptyStr],
            _max_items(MAX_ADDITIONAL_SUBAGENTS, f"must contain at most {MAX_ADDITIONAL_SUBAGENTS} subagents"),
            _unique("subagent IDs must be unique"),
        ]
        | None
    ) = None
    include_extension: IncludeExtension | None = None
    tools: ToolList | None = None

    @model_validator(mode="after")
    def _include_extension_is_exclusive(self) -> "_LevelFields":
        if self.include_extension is not None and (
            self.additional_instructions is not None
            or self.additional_skills is not None
            or self.additional_subagents is not None
        ):
            raise ValueError(
                "includeExtension cannot be combined with additionalInstructions, additionalSkills, "
                "or additionalSubagents at the same level"
            )
        return self


Tags = list[Annotated[str, _non_empty("tag must not be empty")]]


class EvalCaseYAML(_LevelFields):
    id: NonEmptyStr
    tags: Tags | None = None
    expect_pass: StrictBool = True
    turns: Annotated[list[EvalTurn], _min_items(1, "must contain at least one turn")]


class EvalSuiteYAML(_LevelFields):
    """The root eval suite file, `eval/eval.yaml`."""

    include: list[Annotated[str, _non_empty("include path must not be empty")]] | None = None
    cases: list[EvalCaseYAML] | None = None


class EvalSuiteFragmentYAML(_LevelFields):
    """A file listed under `include` in the root eval suite file."""

    tags: Tags | None = None
    cases: Annotated[list[EvalCaseYAML], _min_items(1, "must contain at least one case")]


class SystemAgentExtension(_EvalModel):
    type: Literal["draft", "published"]
    additional_instructions: str = ""
    additional_skills: list[str] = Field(default_factory=list)
    additional_subagents: list[str] = Field(default_factory=list)
    last_updated_owner: str | None = None
    last_updated_time: int | None = None


class _SystemAgentExtensionLimits(_EvalModel):
    additional_instructions: Annotated[str, _max_length(MAX_ADDITIONAL_INSTRUCTIONS)]
    additional_skills: Annotated[list[NonEmptyStr], _unique("skill IDs must be unique")]
    additional_subagents: Annotated[
        list[NonEmptyStr],
        _max_items(MAX_ADDITIONAL_SUBAGENTS, f"must contain at most {MAX_ADDITIONAL_SUBAGENTS} subagents"),
        _unique("subagent IDs must be unique"),
    ]


class AgentProjectConfigFile(_EvalModel):
    """`agent-project-config.yaml`, which has the same shape as `GET /ai/project-config`."""

    default_agent: str | None = None
    system_agent_extensions: list[SystemAgentExtension]


@dataclass
class EvalCase:
    """An eval case after includes, cascading, and session context have been resolved."""

    id: str
    turns: list[EvalTurn]
    source: str
    expect_pass: bool = True
    tags: list[str] | None = None
    context_information: ChatContextInformation | None = None
    tools: list[EvalTool] | None = None
    published_subagents: list[str] | None = None


@dataclass
class _LevelContext:
    """Session context set on one level. `None` means not set, so a less specific level is used."""

    data_models: list[EvalDataModel] | None = None
    instance_spaces: list[str] | None = None
    app_context: str | None = None
    additional_instructions: str | None = None
    additional_skills: list[str] | None = None
    additional_subagents: list[str] | None = None
    published_subagents: list[str] | None = None


@dataclass
class _LoadState:
    directory: Path
    eval_dir: Path
    extension_cache: dict[Path, AgentProjectConfigFile] = field(default_factory=dict)


def format_validation_error(error: ValidationError) -> str:
    issues: list[str] = []
    for issue in error.errors():
        message = str(issue["msg"]).removeprefix("Value error, ")
        location = ".".join(str(part) for part in issue["loc"])
        issues.append(f"{location}: {message}" if location else message)
    return "; ".join(issues)


def display_source(directory: Path, path: Path) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()


def resolve_eval_suite(directory: Path) -> tuple[Path, Path]:
    """Find the root eval suite file for a directory.

    Returns:
        The eval directory (include paths are relative to it) and the root eval suite file.
    """
    direct = directory / EVAL_CASES_FILE
    nested = directory / EVAL_DIR / EVAL_CASES_FILE
    has_direct, has_nested = direct.is_file(), nested.is_file()
    if has_direct and has_nested:
        raise AgentEvalError(
            f"Found both eval suite layouts in:\n  {directory}\n  - {direct}\n  - {nested}",
            hint=f"Keep one eval suite config: either {EVAL_CASES_FILE} in the directory you pass, "
            f"or {_EVAL_SUITE_PATH} for an agent project. Remove the other.",
        )
    if has_direct:
        return directory, direct
    if has_nested:
        return directory / EVAL_DIR, nested

    found = _find_eval_yaml_files_under(directory)
    if len(found) == 1:
        return found[0].parent, found[0]
    if len(found) > 1:
        listed = "\n".join(f"  - {source}" for source in sorted(display_source(directory, path) for path in found))
        raise AgentEvalError(
            f"Found multiple {EVAL_CASES_FILE} files in:\n  {directory}\n{listed}",
            hint="Point [DIR] at the directory that contains the eval suite you want to run.",
        )
    raise AgentEvalError(
        f"No eval suite config found in:\n  {directory}\n  Attempted:\n  - {direct}\n  - {nested}\n"
        f"  Recursive search found no {EVAL_CASES_FILE} under this directory.",
        hint=f"Add {EVAL_CASES_FILE} to the directory you pass, {_EVAL_SUITE_PATH} for an agent project, "
        f"or nest a single {EVAL_CASES_FILE} deeper (for example eval/origem/{EVAL_CASES_FILE}).",
    )


def _find_eval_yaml_files_under(directory: Path) -> list[Path]:
    found: list[Path] = []
    for current, dir_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        dir_names[:] = [
            name
            for name in dir_names
            if not name.startswith(".")
            and name not in _SKIP_WALK_DIRECTORIES
            and not (current_path / name).is_symlink()
        ]
        if EVAL_CASES_FILE in file_names and not (current_path / EVAL_CASES_FILE).is_symlink():
            found.append(current_path / EVAL_CASES_FILE)
    return found


def load_eval_cases(directory: Path) -> list[EvalCase]:
    """Load all eval cases of the eval suite in a directory, including the cases of included files."""
    eval_dir, root_path = resolve_eval_suite(directory)
    state = _LoadState(directory=directory, eval_dir=eval_dir)
    root_source = display_source(directory, root_path)

    root = _validate(EvalSuiteYAML, _read_yaml(root_path), root_path)
    root_context = _materialize_level(root, state)
    cases = [_build_case(case, root_source, root_context, root.tools, state) for case in root.cases or []]

    for include in root.include or []:
        fragment_path = eval_dir / include
        fragment = _validate(EvalSuiteFragmentYAML, _read_yaml(fragment_path), fragment_path)
        fragment_context = _materialize_level(fragment, state)
        source = display_source(directory, fragment_path)
        for case in fragment.cases:
            cases.append(
                _build_case(
                    case,
                    source,
                    root_context,
                    root.tools,
                    state,
                    file_context=fragment_context,
                    file_tags=fragment.tags,
                    file_tools=fragment.tools,
                )
            )

    if not cases:
        raise AgentEvalError("No eval cases found.", hint=f"Add a cases list or an include field to {root_source}.")
    _assert_unique_case_ids(cases)
    _assert_cases_have_scored_turn(cases)
    return cases


def filter_eval_cases(
    cases: list[EvalCase], case_filter: Sequence[str] | None = None, tag_filter: Sequence[str] | None = None
) -> list[EvalCase]:
    selected = cases
    if case_filter:
        selected = [case for case in selected if case.id in case_filter]
    if tag_filter:
        tags = set(tag_filter)
        selected = [case for case in selected if case.tags and any(tag in tags for tag in case.tags)]
    if not selected:
        raise AgentEvalError(
            "No eval cases matched the filter.",
            hint="Check the --case and --tag flag values against the loaded eval suite.",
        )
    return selected


def _read_yaml(path: Path) -> Any:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        raise AgentEvalError(f"Eval file not found:\n  {path}") from None
    try:
        return read_yaml_content(content)
    except YAMLError as error:
        raise AgentEvalError(f"Failed to parse {path}: {error}", hint="Check the YAML syntax of the file.") from None


TModel = TypeVar("TModel", bound=BaseModel)


def _validate(model: type[TModel], data: Any, path: Path) -> TModel:
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise AgentEvalError(f"Invalid {path}: {format_validation_error(error)}") from None


def _materialize_level(level: _LevelFields, state: _LoadState) -> _LevelContext:
    context = _LevelContext(
        data_models=level.data_models, instance_spaces=level.instance_spaces, app_context=level.app_context
    )
    if level.include_extension is not None:
        extension = _load_extension(level.include_extension, state)
        context.additional_instructions = extension.additional_instructions
        context.additional_skills = list(extension.additional_skills)
        context.additional_subagents = list(extension.additional_subagents)
        if level.include_extension.type == "published":
            context.published_subagents = list(extension.additional_subagents)
    else:
        context.additional_instructions = level.additional_instructions
        context.additional_skills = level.additional_skills
        context.additional_subagents = level.additional_subagents
    return context


def _load_extension(include: IncludeExtension, state: _LoadState) -> SystemAgentExtension:
    path = state.eval_dir / include.path
    if path not in state.extension_cache:
        state.extension_cache[path] = _read_project_config(path, state.eval_dir)
    document = state.extension_cache[path]
    extension = next((item for item in document.system_agent_extensions if item.type == include.type), None)
    if extension is None:
        raise AgentEvalError(
            f"Extension file {path} has no {include.type} entry in systemAgentExtensions.",
            hint=f"Add a systemAgentExtensions entry with type: {include.type}.",
        )
    try:
        _SystemAgentExtensionLimits.model_validate(extension.model_dump())
    except ValidationError as error:
        raise AgentEvalError(
            f"Extension {path} ({include.type}): {format_validation_error(error)}",
            hint=f"Edit systemAgentExtensions.{include.type} in that extension file (see the field errors above).",
        ) from None
    return extension


def _read_project_config(path: Path, eval_dir: Path) -> AgentProjectConfigFile:
    hint = (
        "Check includeExtension.path in your eval YAML (paths are relative to the directory that "
        f"contains the root eval.yaml: {eval_dir})."
    )
    if not path.exists():
        raise AgentEvalError(f"Extension file not found:\n  {path}", hint=hint)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise AgentEvalError(f"Cannot read extension file:\n  {path}\n  {error}", hint=hint) from None
    try:
        data = read_yaml_content(content)
    except YAMLError:
        raise AgentEvalError(f"Invalid YAML in {path}.", hint="Fix the YAML syntax and try again.") from None
    try:
        document = AgentProjectConfigFile.model_validate(data)
    except ValidationError:
        raise AgentEvalError(
            f"{path} must be API-shaped project-config (defaultAgent + systemAgentExtensions).",
            hint="Mirror GET …/ai/project-config items[0] (defaultAgent, systemAgentExtensions with type, "
            "additionalInstructions, additionalSkills, additionalSubagents).",
        ) from None
    extensions = document.system_agent_extensions
    if len(extensions) > 2:
        raise AgentEvalError(
            f"{path}: systemAgentExtensions has {len(extensions)} entries; "
            "at most one draft and one published are allowed.",
            hint="Keep at most two entries with distinct type values.",
        )
    if len({item.type for item in extensions}) != len(extensions):
        raise AgentEvalError(
            f"{path}: systemAgentExtensions must not contain duplicate type values.",
            hint="Include at most one entry with type: draft and one with type: published.",
        )
    return document


def _first_set(*values: T | None) -> T | None:
    return next((value for value in values if value is not None), None)


def _cascade(root: _LevelContext, file: _LevelContext | None, case: _LevelContext) -> _LevelContext:
    levels = [case, file, root]
    subagent_level = next(
        (level for level in levels if level is not None and level.additional_subagents is not None), None
    )

    def pick(name: str) -> Any:
        return _first_set(*(getattr(level, name) for level in levels if level is not None))

    return _LevelContext(
        data_models=pick("data_models"),
        instance_spaces=pick("instance_spaces"),
        app_context=pick("app_context"),
        additional_instructions=pick("additional_instructions"),
        additional_skills=pick("additional_skills"),
        additional_subagents=subagent_level.additional_subagents if subagent_level else None,
        published_subagents=subagent_level.published_subagents if subagent_level else None,
    )


def _expand_session_context(context: _LevelContext) -> ChatContextInformation | None:
    parts = [part for part in (context.app_context, context.additional_instructions) if part]
    values: dict[str, Any] = {}
    if context.data_models:
        values["data_models"] = [
            ChatDataModel(space=model.space, external_id=model.external_id, version=model.version)
            for model in context.data_models
        ]
    if context.instance_spaces:
        values["instance_spaces"] = list(context.instance_spaces)
    if parts:
        values["app_context"] = "\n\n".join(parts)
    if context.additional_skills:
        values["skills"] = list(context.additional_skills)
    if context.additional_subagents:
        values["subagents"] = list(context.additional_subagents)
    if not values:
        return None
    return ChatContextInformation(**values)


def _build_case(
    case: EvalCaseYAML,
    source: str,
    root_context: _LevelContext,
    root_tools: list[EvalTool] | None,
    state: _LoadState,
    file_context: _LevelContext | None = None,
    file_tags: list[str] | None = None,
    file_tools: list[EvalTool] | None = None,
) -> EvalCase:
    tags = case.tags
    if file_tags:
        tags = list(dict.fromkeys([*file_tags, *(case.tags or [])]))
    resolved = _cascade(root_context, file_context, _materialize_level(case, state))
    context_information = _expand_session_context(resolved)
    if context_information is not None:
        if context_information.data_models and not context_information.instance_spaces:
            raise AgentEvalError(
                f'Case "{case.id}" declares dataModels without instanceSpaces',
                hint="Add instanceSpaces alongside dataModels in your session context.",
            )
        if context_information.instance_spaces and not context_information.data_models:
            raise AgentEvalError(
                f'Case "{case.id}" declares instanceSpaces without dataModels',
                hint="Add dataModels alongside instanceSpaces in your session context.",
            )
    return EvalCase(
        id=case.id,
        turns=case.turns,
        source=source,
        expect_pass=case.expect_pass,
        tags=tags,
        context_information=context_information,
        tools=_first_set(case.tools, file_tools, root_tools),
        published_subagents=resolved.published_subagents or None,
    )


def _assert_unique_case_ids(cases: list[EvalCase]) -> None:
    sources_by_id: dict[str, list[str]] = {}
    for case in cases:
        sources_by_id.setdefault(case.id, []).append(case.source)
    for case_id, sources in sources_by_id.items():
        if len(sources) <= 1:
            continue
        listed = "\n".join(f"    - {source}" for source in dict.fromkeys(sources))
        raise AgentEvalError(
            f'Duplicate case id "{case_id}"\n  Defined in:\n{listed}', hint="Each case must have a unique id."
        )


def _assert_cases_have_scored_turn(cases: list[EvalCase]) -> None:
    for case in cases:
        if not any(turn.scorers for turn in case.turns):
            raise AgentEvalError(
                f'Case "{case.id}" ({case.source}) has no scored turns',
                hint="Add scorers to at least one turn in each case.",
            )
