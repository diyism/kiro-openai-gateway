# -*- coding: utf-8 -*-

# Kiro OpenAI Gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Pydantic models for Anthropic Messages API compatibility.

Defines data schemas for Claude Code and other Anthropic API clients.
"""

import time
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


# ==================================================================================================
# Content Block Types
# ==================================================================================================

class TextContent(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """Image source data."""
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class ImageContent(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseContent(BaseModel):
    """Tool use content block."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(BaseModel):
    """Tool result content block."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Union[TextContent, ImageContent]]]
    is_error: Optional[bool] = None


ContentBlock = Union[TextContent, ImageContent, ToolUseContent, ToolResultContent]


# ==================================================================================================
# Message Models
# ==================================================================================================

class AnthropicMessage(BaseModel):
    """
    Message in Anthropic format.

    Attributes:
        role: Message role (user or assistant)
        content: Message content (string or list of content blocks)
    """
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    model_config = {"extra": "allow"}


# ==================================================================================================
# Tool Models
# ==================================================================================================

class AnthropicToolInputSchema(BaseModel):
    """JSON Schema for tool input parameters."""
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    """
    Tool definition in Anthropic format.

    Attributes:
        name: Tool name
        description: Tool description
        input_schema: JSON Schema for tool parameters
    """
    name: str
    description: str
    input_schema: AnthropicToolInputSchema


# ==================================================================================================
# Request Model
# ==================================================================================================

class AnthropicMessagesRequest(BaseModel):
    """
    Request for Anthropic Messages API.

    Compatible with Claude Code and other Anthropic API clients.

    Attributes:
        model: Model identifier (e.g., "claude-sonnet-4-20250514")
        messages: List of messages
        max_tokens: Maximum tokens to generate
        system: System prompt (optional)
        temperature: Temperature for sampling (0-1)
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: Stop sequences
        stream: Enable streaming
        tools: Available tools
        tool_choice: Tool choice strategy
    """
    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    max_tokens: int = Field(default=4096, ge=1)

    # Optional parameters
    system: Optional[Union[str, List[TextContent]]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

    # Tools
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[Dict[str, Any], Literal["auto", "any"]]] = None

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Response Models
# ==================================================================================================

class AnthropicUsage(BaseModel):
    """Token usage information."""
    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseModel):
    """
    Response from Anthropic Messages API (non-streaming).

    Attributes:
        id: Unique response ID
        type: Response type ("message")
        role: Assistant role
        content: Response content blocks
        model: Model used
        stop_reason: Reason for stopping
        stop_sequence: Stop sequence that triggered stop (if any)
        usage: Token usage information
    """
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ==================================================================================================
# Streaming Response Models
# ==================================================================================================

class MessageStartEvent(BaseModel):
    """Message start event in streaming."""
    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    """Content block start event."""
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(BaseModel):
    """Content block delta event."""
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    """Content block stop event."""
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Message delta event."""
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None


class MessageStopEvent(BaseModel):
    """Message stop event."""
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """Ping event."""
    type: Literal["ping"] = "ping"


StreamEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent
]


# ==================================================================================================
# Error Models
# ==================================================================================================

class AnthropicError(BaseModel):
    """Error response from Anthropic API."""
    type: Literal["error"] = "error"
    error: Dict[str, Any]
