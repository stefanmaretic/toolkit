"""Chat completions API (`POST /ai/chat/completions`) for calling the language models available in a CDF project.

Note: This is an alpha API and may change in future releases.
"""

from cognite_toolkit._cdf_tk.client.http_client import HTTPClient, RequestMessage
from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import ChatCompletionRequest, ChatCompletionResponse

CHAT_COMPLETIONS_API_VERSION = "20230101-alpha"


class ChatCompletionsAPI:
    def __init__(self, http_client: HTTPClient) -> None:
        self._http_client = http_client

    @property
    def url(self) -> str:
        return self._http_client.config.create_api_url("/ai/chat/completions")

    def create(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Create a (non-streaming) chat completion.

        Args:
            request: The model and messages to send.

        Returns:
            The completion.
        """
        message = RequestMessage(
            endpoint_url=self.url,
            method="POST",
            body_content=request.dump(),
            api_version=CHAT_COMPLETIONS_API_VERSION,
        )
        response = self._http_client.request_single_retries(message).get_success_or_raise(message)
        return ChatCompletionResponse.model_validate_json(response.body)
