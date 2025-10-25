"""Main LLM client for inference with tool support."""

from distutils.util import strtobool
import json
import logging
import re
import time
from copy import copy
from typing import List, Dict, Any, Optional, Callable

import os
from openai import OpenAI
from pydantic import BaseModel

from .utils import lm_json_utils
from .schemas import ToolFunction, ToolCall, ToolResult
from .retry_handler import RetryHandler
from .tool_manager import ToolManager
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class LLM:
    """
    Main LLM client with support for completions, embeddings, and function calling.
    
    Features:
    - Automatic retry with exponential backoff
    - Tool/function calling support
    - Content sanitization for CloudFront WAF compatibility
    - Token management and truncation
    - Conversation history management
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize LLM client.
        
        Args:
            type: 'completion' or 'embedding' (default: 'completion')
            model: Model ID (defaults to INFERENCE_MODEL_ID env var)
            temperature: Sampling temperature (default: 0.15)
            frequency_penalty: Frequency penalty (default: 0.2)
            presence_penalty: Presence penalty (default: 0.2)
            response_format: Pydantic model for structured output
            continuous: Keep conversation history (default: True)
            messages: Initial conversation history
            max_length: Maximum response length (default: 100,000)
            input_tokens_limit: Maximum input tokens (default: 40,000)
            max_retry: Maximum retry attempts (default: 3)
            system: System prompt
            enable_tools: Enable tool calling (default: False)
            sanitize_tool_responses: Sanitize tool responses (default: True)
            sanitization_patterns: Custom sanitization patterns
            max_tool_response_tokens: Max tokens in tool responses (default: 4000)
            track_tool_metadata: Track tool execution metadata (default: False)
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Validate required environment variables
        self._validate_environment()
        
        # Basic configuration
        self.type = kwargs.get('type', 'completion')
        if self.type not in ['completion', 'embedding']:
            raise ValueError("type must be 'completion' or 'embedding'")
        
        if self.type == 'embedding':
            raise Exception("Embedding model is not supported yet. Use 'completion' type for now.")
        
        # Initialize OpenAI client
        self.base_url = os.environ['INFERENCE_URL']
        self.client = OpenAI(
            api_key=os.environ['INFERENCE_KEY'],
            base_url=self.base_url
        )
        
        # Model configuration
        self.model = kwargs.get('model', os.environ['INFERENCE_MODEL_ID'])
        self.temperature = kwargs.get('temperature', 0.15)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.2)
        self.presence_penalty = kwargs.get('presence_penalty', 0.2)
        
        # Response format (structured output)
        self.response_format = kwargs.get('response_format', None)
        if self.response_format and not issubclass(self.response_format, BaseModel):
            raise ValueError("response_format must be a subclass of BaseModel (pydantic)")
        
        # Conversation management
        self.continuous = kwargs.get('continuous', True)
        self.messages = kwargs.get('messages', [])
        self.max_length = kwargs.get('max_length', 100_000)
        self.input_tokens_limit = kwargs.get('input_tokens_limit', 40_000)
        self.max_retry = kwargs.get('max_retry', 3)
        self.retry_count = 0
        self.system = kwargs.get('system', None)
        
        # Initialize handlers
        self.retry_handler = RetryHandler(max_retries=self.max_retry)
        self.tool_manager = ToolManager()
        self.content_processor = ContentProcessor(
            sanitize_enabled=kwargs.get('sanitize_tool_responses', True),
            sanitization_patterns=kwargs.get('sanitization_patterns', None),
            max_tokens=kwargs.get('max_tool_response_tokens', 4000),
            track_metadata=kwargs.get('track_tool_metadata', False)
        )
        
        # Tool configuration
        self.enable_tools = kwargs.get('enable_tools', False)
        self.tool_request_delay = float(os.environ.get('LLM_TOOL_REQUEST_DELAY', '0'))
        
        # Backward compatibility - maintain tools dict reference
        self.tools = self.tool_manager.tools
        
        # Add system message if provided
        if self.system:
            self.add_message("system", self.system)
    
    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = ['INFERENCE_URL', 'INFERENCE_MODEL_ID', 'INFERENCE_KEY']
        for var in required_vars:
            if not os.environ.get(var):
                raise ValueError(f"{var} environment variable must be set.")
    
    # ========== Model Capability Properties ==========
    
    @property
    def supports_parallel_requests(self) -> bool:
        """Check if model supports parallel requests."""
        return strtobool(os.environ.get('INFERENCE_SUPPORTS_PARALLEL_REQUESTS', 'True'))
    
    @property
    def supports_frequency_penalty(self) -> bool:
        """Check if model supports frequency penalty."""
        return 'claude' not in self.model.lower() and 'gemini' not in self.model.lower()
    
    @property
    def supports_presence_penalty(self) -> bool:
        """Check if model supports presence penalty."""
        return 'claude' not in self.model.lower() and 'gemini' not in self.model.lower()
    
    @property
    def require_response_format(self) -> bool:
        """Check if response format is required."""
        return self.response_format is not None and issubclass(self.response_format, BaseModel)
    
    # ========== Message Management ==========
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content
            
        Raises:
            ValueError: If role is invalid
        """
        if not isinstance(role, str) or role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        
        content = self._sanitize_messages(content)
        
        if self.require_response_format:
            content = (
                f"{content}\n"
                f"Here is the JSON schema you need to follow for the response: "
                f"{self.response_format.schema_json(indent=2)}\n"
                f"Do not return the entire schema, only the response in JSON format. "
                f"Use the schema only as a guide for filling in.\n"
            )
        
        self.messages.append({"role": role, "content": content})
    
    def _sanitize_messages(self, content: str) -> str:
        """Remove excessive whitespace from messages."""
        return self.content_processor.sanitize_messages(content)
    
    def reset(self):
        """Reset conversation history and retry counter."""
        self.messages = []
        self.retry_count = 0
        if self.system:
            self.add_message("system", self.system)
    
    # ========== API Request Management ==========
    
    def get_request_params(self) -> Dict[str, Any]:
        """Build parameters for completion request."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
        }
        
        if self.supports_frequency_penalty:
            params['frequency_penalty'] = self.frequency_penalty
        
        if self.supports_presence_penalty:
            params['presence_penalty'] = self.presence_penalty
        
        # Add tools if enabled
        if self.enable_tools and self.tool_manager.tools:
            params['tools'] = self.tool_manager.get_tools_schema()
            params['tool_choice'] = 'auto'
        
        return params
    
    def get_embedding_params(self) -> Dict[str, Any]:
        """Build parameters for embedding request."""
        return {"model": self.model}
    
    # ========== Core API Methods ==========
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Text cannot be empty for embedding.")
        
        params = self.get_embedding_params()
        logger.info(f"Embedding text with parameters: {params}")
        
        response = self.retry_handler.execute_with_retry(
            self.client.embeddings.create,
            **params,
            input=text
        )
        
        return response.data[0].embedding if response.data else None
    
    def prompt(self, prompt: Optional[str] = None, system: Optional[str] = None, retry: bool = False) -> str:
        """
        Generate completion from prompt.
        
        Args:
            prompt: User prompt
            system: Optional system message
            retry: Internal retry flag
            
        Returns:
            Generated response text or structured output
        """
        if prompt and not retry:
            self.retry_count = 0
            if system:
                self.add_message("system", system)
            self.add_message("user", prompt)
        
        params = self.get_request_params()
        logger.info(f"Prompting LLM with parameters: {params}")
        
        # Validate messages before sending
        validated_messages = self._validate_messages_for_api(self.messages)
        
        response = self.retry_handler.execute_with_retry(
            self.client.chat.completions.create,
            **params,
            messages=validated_messages
        )
        
        result = self.process_response(response)
        
        if not self.continuous:
            self.reset()
        
        return result
    
    def retry_prompt(self, prompt: Optional[str] = None) -> Optional[str]:
        """
        Retry the last prompt.
        
        Args:
            prompt: Optional new prompt
            
        Returns:
            Response or None if max retries exceeded
        """
        if self.retry_count >= self.max_retry:
            logger.warning(f"Maximum retry limit ({self.max_retry}) reached. Giving up on retry.")
            return None
        
        self.retry_count += 1
        logger.info(f"Retrying last prompt (attempt {self.retry_count}/{self.max_retry}).")
        return self.prompt(prompt, retry=True)
    
    # ========== Response Processing ==========
    
    def process_response(self, response) -> str:
        """
        Process LLM response, handling tool calls if present.
        
        Args:
            response: OpenAI API response
            
        Returns:
            Processed response text or structured output
        """
        choice = response.choices[0]
        
        # Handle tool calls
        if (hasattr(choice.message, 'tool_calls') and 
            choice.message.tool_calls is not None):
            try:
                if hasattr(choice.message.tool_calls, '__iter__') and len(choice.message.tool_calls) > 0:
                    return self._handle_tool_calls(choice.message, response)
            except (TypeError, AttributeError):
                pass  # Mock object, skip tool handling
        
        # Handle regular message
        if not choice.message.content:
            logger.warning("Received empty response from LLM")
            return ""
        
        message = choice.message.content.strip()
        
        # Remove thinking tags
        message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)
        
        # Extract response tags if present
        if '<response>' in message and '</response>' in message:
            message = re.search(r'<response>(.*?)</response>', message, flags=re.DOTALL).group(1)
        
        # Check max length
        if len(message) > self.max_length:
            logger.warning(f"Max length exceeded: {len(message)} > {self.max_length}")
            retry_result = self.retry_prompt()
            if retry_result is not None:
                return retry_result
        
        self.add_message("assistant", message)
        
        # Handle structured output
        if self.require_response_format:
            logger.info(f"Original message: {message}")
            message = lm_json_utils.extract_strict_json(message)
            logger.info(f"Parsed message: {message}")
            return self.response_format.parse_raw(message)
        
        return response.choices[0].message.content
    
    def _handle_tool_calls(self, message, original_response):
        """
        Execute tool calls and get follow-up response.
        
        Args:
            message: Message with tool calls
            original_response: Original API response
            
        Returns:
            Final response after tool execution
        """
        # Build assistant message with tool calls
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        }
        
        # Handle content
        try:
            raw_content = message.content if hasattr(message, 'content') else None
        except Exception:
            raw_content = None
        
        if raw_content and str(raw_content).strip():
            assistant_message["content"] = str(raw_content).strip()
        else:
            assistant_message["content"] = "[Tool calls in progress]"
        
        self.messages.append(assistant_message)
        
        # Execute tool calls
        start_time = time.time()
        
        for tool_call in message.tool_calls:
            tool_start_time = time.time()
            
            try:
                # Parse and execute
                arguments = json.loads(tool_call.function.arguments)
                tc = ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments
                )
                
                result = self.tool_manager.execute_tool(tc)
                
                # Process response with content processor
                execution_time = time.time() - tool_start_time
                processed = self.content_processor.process_tool_response(
                    result.content,
                    tool_call.function.name,
                    execution_time=execution_time
                )
                
                # Build tool message
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": processed['content']
                }
                
                if 'metadata' in processed:
                    tool_message['metadata'] = processed['metadata']
                
                self.messages.append(tool_message)
                
                # Log execution
                if 'metadata' in processed:
                    meta = processed['metadata']
                    logger.info(
                        f"Executed tool {tool_call.function.name}: "
                        f"tokens={meta['final_tokens']}, "
                        f"time={meta.get('execution_time_ms', 0)}ms, "
                        f"sanitized={meta['was_sanitized']}, "
                        f"truncated={meta['was_truncated']}"
                    )
                else:
                    logger.info(f"Executed tool {tool_call.function.name}")
                
            except Exception as e:
                logger.error(f"Error processing tool call {tool_call.function.name}: {str(e)}")
                
                processed = self.content_processor.process_tool_response(
                    f"Error: {str(e)}",
                    tool_call.function.name
                )
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": processed['content']
                })
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(f"Executed {len(message.tool_calls)} tool(s) in {total_time*1000:.0f}ms")
        
        # Add delay before follow-up
        if self.tool_request_delay > 0:
            logger.info(f"Waiting {self.tool_request_delay}s before follow-up request")
            time.sleep(self.tool_request_delay)
        
        # Get follow-up response
        params = self.get_request_params()
        validated_messages = self._validate_messages_for_api(self.messages)
        
        follow_up_response = self.retry_handler.execute_with_retry(
            self.client.chat.completions.create,
            **params,
            messages=validated_messages
        )
        
        return self.process_response(follow_up_response)
    
    def _validate_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """
        Validate and sanitize messages for API.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Validated messages
        """
        validated_messages = []
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Skipping invalid message type at index {i}: {type(msg)}")
                continue
            
            # Copy message
            validated_msg = dict(msg)
            
            # Validate role
            if 'role' not in validated_msg or not validated_msg['role']:
                logger.warning(f"Skipping message at index {i} without role")
                continue
            
            validated_msg['role'] = str(validated_msg['role']).strip()
            if not validated_msg['role']:
                logger.warning(f"Skipping message at index {i} with empty role")
                continue
            
            # Validate content
            if 'content' not in validated_msg:
                validated_msg['content'] = "Message content unavailable"
            elif validated_msg['content'] is None:
                validated_msg['content'] = "Message content unavailable"
            
            # Convert to string
            try:
                validated_msg['content'] = str(validated_msg['content'])
            except Exception as e:
                logger.error(f"Failed to convert content to string for message {i}: {e}")
                validated_msg['content'] = "Message content conversion failed"
            
            # Apply role-specific fixes
            role = validated_msg['role'].lower()
            content = validated_msg['content'].strip()
            
            if not content:
                if role == 'tool':
                    validated_msg['content'] = "Tool executed successfully"
                elif role == 'assistant' and 'tool_calls' in validated_msg:
                    validated_msg['content'] = " "
                else:
                    validated_msg['content'] = "Content unavailable"
            else:
                # Fix tool content if it's a placeholder
                if role == 'tool' and content in ["Message content unavailable", "Content unavailable"]:
                    validated_msg['content'] = "Tool executed successfully"
            
            # Ensure no empty content
            if validated_msg['content'] == "":
                validated_msg['content'] = " " if (role == 'assistant' and 'tool_calls' in validated_msg) else "."
            
            validated_messages.append(validated_msg)
        
        return validated_messages
    
    # ========== Tool Management (Delegated) ==========
    
    def add_tool(self, tool: ToolFunction):
        """Add a tool function."""
        self.tool_manager.add_tool(tool)
    
    def remove_tool(self, name: str):
        """Remove a tool by name."""
        self.tool_manager.remove_tool(name)
    
    def add_tool_from_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        """Add a tool from a Python function."""
        self.tool_manager.add_tool_from_function(func, name, description, parameters)
    
    def list_tools(self) -> List[str]:
        """Get list of registered tools."""
        return self.tool_manager.list_tools()
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        return self.tool_manager.execute_tool(tool_call)
    
    def enable_tool_support(self, enabled: bool = True):
        """Enable or disable tool support."""
        self.enable_tools = enabled
        logger.info(f"Tool support {'enabled' if enabled else 'disabled'}")
    
    def clear_tools(self):
        """Clear all registered tools."""
        self.tool_manager.clear_tools()
    
    # ========== Backward Compatibility ==========
    
    def _sanitize_tool_response(self, content: str) -> str:
        """Sanitize tool response (backward compatibility)."""
        return self.content_processor.sanitize_content(content)
    
    def _truncate_tool_response(self, content: str) -> tuple:
        """Truncate tool response (backward compatibility)."""
        return self.content_processor.truncate_content(content)
    
    def _process_tool_response(self, content: str, tool_name: str) -> dict:
        """Process tool response (backward compatibility)."""
        return self.content_processor.process_tool_response(content, tool_name)
    
    def _api_call_with_retry(self, api_func, *args, **kwargs):
        """Execute API call with retry (backward compatibility)."""
        return self.retry_handler.execute_with_retry(api_func, *args, **kwargs)
    
    def _sanitize_tool_content(self, content: str, tool_name: str, is_error: bool = False) -> str:
        """Sanitize tool content (backward compatibility)."""
        return self.content_processor.sanitize_tool_content(content, tool_name, is_error)
    
    # Expose these properties for backward compatibility
    @property
    def sanitize_tool_responses(self) -> bool:
        """Check if tool response sanitization is enabled."""
        return self.content_processor.sanitize_enabled
    
    @property
    def max_tool_response_tokens(self) -> int:
        """Get maximum tool response tokens."""
        return self.content_processor.max_tokens
    
    @property
    def track_tool_metadata(self) -> bool:
        """Check if tool metadata tracking is enabled."""
        return self.content_processor.track_metadata
