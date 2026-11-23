import json

import httpx2
import pytest
import respx

from cognite_toolkit._cdf_tk.client import ToolkitClient, ToolkitClientConfig
from cognite_toolkit._cdf_tk.client.resource_classes.agent_chat import (
    AgentChatRequest,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatContextInformation,
    ChatTextContent,
    ToolConfirmationChatMessage,
    UserChatMessage,
)

CHAT_RESPONSE = {
    "agentExternalId": "my_agent",
    "response": {
        "type": "result",
        "cursor": "c1",
        "messages": [{"role": "agent", "content": {"type": "text", "text": "Hello"}, "futureField": 1}],
    },
}


@pytest.mark.usefixtures("disable_gzip")
class TestAgentChat:
    def test_chat_posts_camel_case_body_to_public_route(
        self, toolkit_config: ToolkitClientConfig, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(toolkit_config.create_api_url("/ai/agents/chat")).mock(
            return_value=httpx2.Response(200, json=CHAT_RESPONSE)
        )
        request = AgentChatRequest(
            agent_external_id="my_agent",
            messages=[UserChatMessage(content=ChatTextContent(text="Hi"))],
            context_information=ChatContextInformation(instance_spaces=["sp"], app_context="Plant A"),
        )

        response = ToolkitClient(toolkit_config).agents.chat(request)

        assert json.loads(route.calls[0].request.content) == {
            "agentExternalId": "my_agent",
            "messages": [{"role": "user", "content": {"type": "text", "text": "Hi"}}],
            "contextInformation": {"instanceSpaces": ["sp"], "appContext": "Plant A"},
        }
        assert route.calls[0].request.headers["cdf-version"] == "20230101-alpha"
        assert response.response is not None and response.response.cursor == "c1"
        assert response.dump() == CHAT_RESPONSE

    def test_chat_internal_route_and_local_base_url(
        self, toolkit_config: ToolkitClientConfig, respx_mock: respx.MockRouter
    ) -> None:
        client = ToolkitClient(toolkit_config)
        internal = respx_mock.post(toolkit_config.create_api_url("/ai/internal/agents/chat")).mock(
            return_value=httpx2.Response(200, json=CHAT_RESPONSE)
        )
        local_url = f"http://127.0.0.1:5000/api/v1/projects/{toolkit_config.project}/ai/agents/chat"
        local = respx_mock.post(local_url).mock(return_value=httpx2.Response(200, json=CHAT_RESPONSE))
        request = AgentChatRequest(
            agent_external_id="cdf_industrial_v4",
            cursor="c1",
            messages=[ToolConfirmationChatMessage(action_id="a1")],
        )

        client.agents.chat(request, internal=True)
        client.agents.chat(request, base_url="http://127.0.0.1:5000/")

        assert json.loads(internal.calls[0].request.content) == {
            "agentExternalId": "cdf_industrial_v4",
            "cursor": "c1",
            "messages": [{"role": "action", "type": "toolConfirmation", "actionId": "a1", "status": "ALLOW"}],
        }
        assert local.called
        assert client.agents.chat_url(base_url="http://127.0.0.1:5000") == local_url


@pytest.mark.usefixtures("disable_gzip")
class TestChatCompletions:
    def test_create(self, toolkit_config: ToolkitClientConfig, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(toolkit_config.create_api_url("/ai/chat/completions")).mock(
            return_value=httpx2.Response(
                200,
                json={
                    "model": "azure/gpt-5",
                    "choices": [{"message": {"role": "assistant", "content": "{}"}, "finishReason": "stop"}],
                },
            )
        )
        request = ChatCompletionRequest(
            model="azure/gpt-5",
            messages=[
                ChatCompletionMessage(role="system", content="S"),
                ChatCompletionMessage(role="user", content="U"),
            ],
        )

        response = ToolkitClient(toolkit_config).chat_completions.create(request)

        assert json.loads(route.calls[0].request.content) == {
            "model": "azure/gpt-5",
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            "stream": False,
        }
        assert route.calls[0].request.headers["cdf-version"] == "20230101-alpha"
        assert response.first_content() == "{}"
