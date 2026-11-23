from pathlib import Path
from typing import Any

import typer

from cognite_toolkit._cdf_tk.apps._helpers import print_help_if_no_subcommand
from cognite_toolkit._cdf_tk.commands import AgentEvalCommand
from cognite_toolkit._cdf_tk.commands.agent_eval._report import DEFAULT_REPORT_DIR
from cognite_toolkit._cdf_tk.commands.auth import EnvironmentVariables


class AgentsApp(typer.Typer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.callback(invoke_without_command=True)(self.main)
        self.command("eval")(self.eval)

    @staticmethod
    def main(ctx: typer.Context) -> None:
        """Commands to work with Atlas AI agents"""
        print_help_if_no_subcommand(ctx)

    @staticmethod
    def eval(
        directory: Path | None = typer.Argument(
            None,
            help="Agent directory, with <name>.agent.yaml and eval/eval.yaml. Defaults to the current directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
        case: list[str] | None = typer.Option(
            None, "--case", help="Run only cases with this id (repeatable)", show_default=False
        ),
        tag: list[str] | None = typer.Option(
            None, "--tag", help="Run only cases with this tag (repeatable)", show_default=False
        ),
        judge_model: str | None = typer.Option(
            None, "--judge-model", help="LLM model for scoring eval cases (e.g. azure/gpt-5)"
        ),
        external_id: str | None = typer.Option(
            None, "--external-id", help="Agent external ID (alternative to reading from *.agent.yaml)"
        ),
        system_agent: bool = typer.Option(
            False, "--system-agent", help="Evaluate the Cognite system agent (currently cdf_industrial_v4)"
        ),
        upsert: bool = typer.Option(False, "--upsert", help="Push the local *.agent.yaml to CDF before running evals"),
        fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop after the first failing case; skip the rest"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Log request/response traces to stderr"),
        fail_on_tool_confirmation: bool = typer.Option(
            False,
            "--fail-on-tool-confirmation",
            help="Fail a case instead of auto-approving when the agent requests tool confirmation",
        ),
        report_dir: Path | None = typer.Option(
            None,
            "--report-dir",
            help=f"Write structured eval reports (report.json + report.md) to this directory, e.g. {DEFAULT_REPORT_DIR}",
            file_okay=False,
            dir_okay=True,
        ),
    ) -> None:
        """Run eval cases for an agent

        Runs the cases in eval/eval.yaml against a deployed Atlas AI agent and scores the answers with an LLM
        judge. The suite format, the scoring and the report are the same as `cognite agents eval` in @cognite/cli,
        so the same suite can be run with either tool. The command exits with code 1 if any case does not meet
        expectations.

        \b
        Examples:
          Run the eval suite in the current agent directory:
          cdf agents eval

        \b
          Push the local agent first, and write a report to eval-report/:
          cdf agents eval my-agent --upsert --report-dir eval-report

        \b
          Run one case with request/response traces:
          cdf agents eval --case greeting -v

        \b
          Run the cases tagged "ci" against a deployed agent:
          cdf agents eval --tag ci --external-id my_agent
        """
        client = EnvironmentVariables.create_from_environment().get_client()
        cmd = AgentEvalCommand()
        result = cmd.run(
            lambda: cmd.eval(
                client,
                directory=directory,
                case_filter=case,
                tag_filter=tag,
                judge_model=judge_model,
                external_id=external_id,
                system_agent=system_agent,
                upsert=upsert,
                fail_fast=fail_fast,
                verbose=verbose,
                fail_on_tool_confirmation=fail_on_tool_confirmation,
                report_dir=report_dir,
            )
        )
        if result.has_failures:
            raise typer.Exit(code=1)
