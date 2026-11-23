"""Request and response classes for the Atlas AI agent chat and chat completions endpoints.

Based on the API specification at:
https://api-docs.cognite.com/20230101-alpha/tag/Agents

The request and response shapes mirror what `cognite agents eval` in @cognite/cli sends and reads.

Note: These are alpha APIs and may change in future releases. The response classes allow extra fields
so that new fields returned by the API are kept.
"""

from typing import Literal

from pydantic import Field, JsonValue

from cognite_toolkit._cdf_tk.client._resource_base import BaseModelObject


class ChatDataModel(BaseModelObject):
    space: str
    external_id: str
    version: str


class ChatContextInformation(BaseModelObject):
    """Runtime context sent with an agent chat request."""

    data_models: list[ChatDataModel] | None = None
    instance_spaces: list[str] | None = None
    app_context: str | None = None
    skills: list[str] | None = None
    subagents: list[str] | None = None


class ChatTextContent(BaseModelObject):
    type: Literal["text"] = "text"
    text: str


class UserChatMessage(BaseModelObject):
    role: Literal["user"] = "user"
    content: ChatTextContent


class ToolConfirmationChatMessage(BaseModelObject):
    """Answer to a tool confirmation action requested by the agent."""

    role: Literal["action"] = "action"
    type: Literal["toolConfirmation"] = "toolConfirmation"
    action_id: str
    status: Literal["ALLOW", "DENY"] = "ALLOW"


class UserSessionChatMessage(BaseModelObject):
    """Answer to a user session action requested by the agent. The nonce comes from the Sessions API."""

    role: Literal["action"] = "action"
    type: Literal["userSession"] = "userSession"
    action_id: str
    nonce: str


ChatMessage = UserChatMessage | ToolConfirmationChatMessage | UserSessionChatMessage


class AgentChatRequest(BaseModelObject):
    agent_external_id: str
    cursor: str | None = None
    messages: list[ChatMessage]
    context_information: ChatContextInformation | None = None


class AgentToolCall(BaseModelObject):
    name: str | None = None
    input: JsonValue = None
    result: JsonValue = None


class AgentReasoningData(BaseModelObject):
    tool_call: AgentToolCall | None = None


class AgentReasoningContent(BaseModelObject):
    text: str | None = None


class AgentReasoning(BaseModelObject):
    content: list[AgentReasoningContent] | None = None
    data: list[AgentReasoningData] | None = None


class AgentMessageContent(BaseModelObject):
    type: str | None = None
    text: str | None = None


class AgentToolConfirmation(BaseModelObject):
    tool_name: str | None = None
    tool_arguments: JsonValue = None


class AgentAction(BaseModelObject):
    type: str | None = None
    action_id: str | None = None
    id: str | None = None
    tool_confirmation: AgentToolConfirmation | None = None


class AgentResponseMessage(BaseModelObject):
    role: str | None = None
    content: AgentMessageContent | None = None
    reasoning: list[AgentReasoning] | None = None
    actions: list[AgentAction] | None = None


class AgentChatResult(BaseModelObject):
    type: str | None = None
    cursor: str | None = None
    messages: list[AgentResponseMessage] | None = None


class AgentChatResponse(BaseModelObject):
    agent_external_id: str | None = None
    response: AgentChatResult | None = None


class ChatCompletionMessage(BaseModelObject):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModelObject):
    model: str
    messages: list[ChatCompletionMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ChatCompletionResponseMessage(BaseModelObject):
    role: str | None = None
    content: str | None = None


class ChatCompletionChoice(BaseModelObject):
    message: ChatCompletionResponseMessage | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModelObject):
    model: str | None = None
    choices: list[ChatCompletionChoice] = Field(default_factory=list)

    def first_content(self) -> str:
        if not self.choices or self.choices[0].message is None:
            return ""
        return self.choices[0].message.content or ""
