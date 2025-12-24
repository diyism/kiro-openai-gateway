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
Converters for Anthropic Messages API format <-> Kiro API format.

Handles conversion between Anthropic's native format (used by Claude Code)
and Kiro's internal format.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from kiro_gateway.anthropic_models import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
    TextContent,
    ToolUseContent,
    ToolResultContent,
)
from kiro_gateway.config import get_internal_model_id


def extract_text_from_content(content: Any) -> str:
    """
    Extract text from Anthropic content format.

    Args:
        content: Content in Anthropic format (string or list of blocks)

    Returns:
        Extracted text
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
        return "".join(text_parts)

    return str(content)


def extract_tool_uses_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extract tool uses from Anthropic content format.

    Args:
        content: Content in Anthropic format

    Returns:
        List of tool uses in Kiro format
    """
    tool_uses = []

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_uses.append({
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                    "toolUseId": block.get("id", "")
                })
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_uses.append({
                    "name": block.name,
                    "input": block.input,
                    "toolUseId": block.id
                })

    return tool_uses


def extract_tool_results_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extract tool results from Anthropic content format.

    Args:
        content: Content in Anthropic format

    Returns:
        List of tool results in Kiro format
    """
    tool_results = []

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_text = extract_text_from_content(result_content)
                else:
                    result_text = str(result_content)

                tool_results.append({
                    "content": [{"text": result_text}],
                    "status": "error" if block.get("is_error") else "success",
                    "toolUseId": block.get("tool_use_id", "")
                })
            elif hasattr(block, "type") and block.type == "tool_result":
                result_content = block.content
                if isinstance(result_content, list):
                    result_text = extract_text_from_content(result_content)
                else:
                    result_text = str(result_content)

                tool_results.append({
                    "content": [{"text": result_text}],
                    "status": "error" if getattr(block, "is_error", False) else "success",
                    "toolUseId": block.tool_use_id
                })

    return tool_results


def build_kiro_history_from_anthropic(
    messages: List[AnthropicMessage],
    model_id: str
) -> List[Dict[str, Any]]:
    """
    Build Kiro history from Anthropic messages.

    Args:
        messages: List of Anthropic messages
        model_id: Internal Kiro model ID

    Returns:
        List of history entries in Kiro format
    """
    history = []

    for msg in messages:
        if msg.role == "user":
            text_content = extract_text_from_content(msg.content)

            user_input = {
                "content": text_content,
                "modelId": model_id,
                "origin": "AI_EDITOR",
            }

            # Check for tool results
            tool_results = extract_tool_results_from_content(msg.content)
            if tool_results:
                user_input["userInputMessageContext"] = {"toolResults": tool_results}

            history.append({"userInputMessage": user_input})

        elif msg.role == "assistant":
            text_content = extract_text_from_content(msg.content)

            assistant_response = {"content": text_content}

            # Check for tool uses
            tool_uses = extract_tool_uses_from_content(msg.content)
            if tool_uses:
                assistant_response["toolUses"] = tool_uses

            history.append({"assistantResponseMessage": assistant_response})

    return history


def convert_anthropic_tools_to_kiro(tools: Optional[List[AnthropicTool]]) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tools to Kiro format.

    Args:
        tools: List of Anthropic tools

    Returns:
        List of tools in Kiro format or None
    """
    if not tools:
        return None

    kiro_tools = []
    for tool in tools:
        # Convert input_schema to dict if it's a Pydantic model
        if hasattr(tool.input_schema, "model_dump"):
            input_schema = tool.input_schema.model_dump(exclude_none=True)
        else:
            input_schema = tool.input_schema

        kiro_tools.append({
            "toolSpecification": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": input_schema}
            }
        })

    return kiro_tools


def build_kiro_payload_from_anthropic(
    request_data: AnthropicMessagesRequest,
    conversation_id: str,
    profile_arn: str
) -> Dict[str, Any]:
    """
    Build Kiro API payload from Anthropic Messages API request.

    Args:
        request_data: Anthropic Messages API request
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Kiro API payload

    Raises:
        ValueError: If no messages provided
    """
    if not request_data.messages:
        raise ValueError("No messages provided")

    # Get internal model ID
    model_id = get_internal_model_id(request_data.model)

    # Extract system prompt
    system_prompt = ""
    if request_data.system:
        if isinstance(request_data.system, str):
            system_prompt = request_data.system
        elif isinstance(request_data.system, list):
            system_prompt = extract_text_from_content(request_data.system)

    # Build history (all messages except the last one)
    messages = list(request_data.messages)
    history_messages = messages[:-1] if len(messages) > 1 else []

    # Add system prompt to first user message in history if present
    if system_prompt and history_messages:
        first_msg = history_messages[0]
        if first_msg.role == "user":
            original_content = extract_text_from_content(first_msg.content)
            # Create a modified copy
            first_msg = AnthropicMessage(
                role=first_msg.role,
                content=f"{system_prompt}\n\n{original_content}"
            )
            history_messages[0] = first_msg

    history = build_kiro_history_from_anthropic(history_messages, model_id)

    # Current message (last one)
    current_message = messages[-1]
    current_content = extract_text_from_content(current_message.content)

    # If system prompt exists but no history, add to current message
    if system_prompt and not history:
        current_content = f"{system_prompt}\n\n{current_content}"

    # If current message is assistant, add to history and create "Continue" user message
    if current_message.role == "assistant":
        history.append({
            "assistantResponseMessage": {
                "content": current_content
            }
        })
        current_content = "Continue"

    # Ensure content is not empty
    if not current_content:
        current_content = "Continue"

    # Build user input message
    user_input_message = {
        "content": current_content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }

    # Add tools and tool results if present
    user_input_context = {}

    # Add tools
    if request_data.tools:
        kiro_tools = convert_anthropic_tools_to_kiro(request_data.tools)
        if kiro_tools:
            user_input_context["tools"] = kiro_tools

    # Add tool results from current message
    if current_message.role == "user":
        tool_results = extract_tool_results_from_content(current_message.content)
        if tool_results:
            user_input_context["toolResults"] = tool_results

    if user_input_context:
        user_input_message["userInputMessageContext"] = user_input_context

    # Build final payload
    payload = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": user_input_message
            }
        }
    }

    # Add history if not empty
    if history:
        payload["conversationState"]["history"] = history

    # Add profile ARN
    if profile_arn:
        payload["profileArn"] = profile_arn

    return payload
