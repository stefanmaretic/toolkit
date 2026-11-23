from typing import Any

from cognite_toolkit._cdf_tk.exceptions import ToolkitError


class AgentEvalError(ToolkitError):
    """An error raised by `cdf agents eval`.

    The message and hint are kept apart so that they can be written separately to the eval report,
    the same way as `cognite agents eval` in @cognite/cli does.
    """

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        http_status: int | None = None,
        request_url: str | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.http_status = http_status
        self.request_url = request_url
        self.response_body = response_body

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message
