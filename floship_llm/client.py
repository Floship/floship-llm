from distutils.util import strtobool

from copy import copy
import json
import logging
import re
import time
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APIStatusError
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable

from .utils import lm_json_utils # extract_and_fix_json, strict_json
from .schemas import ToolFunction, ToolCall, ToolResult

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, *args, **kwargs):
        # check INFERENCE_URL, INFERENCE_MODEL_ID, INFERENCE_KEY to be present
        if not os.environ.get('INFERENCE_URL'):
            raise ValueError("INFERENCE_URL environment variable must be set.")
        if not os.environ.get('INFERENCE_MODEL_ID'):
            raise ValueError("INFERENCE_MODEL_ID environment variable must be set.")
        if not os.environ.get('INFERENCE_KEY'):
            raise ValueError("INFERENCE_KEY environment variable must be set.")
        
        self.type = kwargs.get('type', 'completion')  # 'completion' or 'embedding'
        if self.type not in ['completion', 'embedding']:
            raise ValueError("type must be 'completion' or 'embedding'")
        self.base_url = os.environ.get('INFERENCE_URL')
        self.client = OpenAI(api_key=os.environ['INFERENCE_KEY'], base_url=self.base_url)
        if self.type == 'embedding':
            # self.model = kwargs.get('model', os.environ.get('EMBEDDING_MODEL'))
            raise Exception("Embedding model is not supported yet. Use 'completion' type for now.")
        elif self.type == 'completion':
            self.model = kwargs.get('model', os.environ.get('INFERENCE_MODEL_ID'))
        self.temperature = kwargs.get('temperature', 0.15)
        # self.max_tokens = kwargs.get('max_tokens', os.environ.get('LARGE_MAX_TOKENS', 1000))
        # self.top_p = kwargs.get('top_p', 1.0)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.2)
        self.presence_penalty = kwargs.get('presence_penalty', 0.2)
        self.response_format = kwargs.get('response_format', None) # Should be a subclass of BaseModel (pydantic)
        self.continuous = kwargs.get('continuous', True) # Will keep conversation history to allow for multi-turn conversations
        if self.response_format and not issubclass(self.response_format, BaseModel):
            raise ValueError("response_format must be a subclass of BaseModel (pydantic)")
        self.messages = kwargs.get('messages', []) # Conversation history, list of dicts with 'role' and 'content'
        self.max_length = kwargs.get('max_length', 100_000) # If set, will retry if the response exceeds this length
        self.input_tokens_limit = kwargs.get('input_tokens_limit', 40_000) # If set, will trim input messages to fit within this limit
        self.max_retry = kwargs.get('max_retry', 3) # Maximum number of retries before giving up
        self.retry_count = 0 # Current retry count
        self.system = kwargs.get('system', None) # System prompt to set the context for the conversation
        
        # Tool support
        self.tools: Dict[str, ToolFunction] = {} # Available tools/functions
        self.enable_tools = kwargs.get('enable_tools', False) # Whether to enable tool calls
        
        if self.system:
            self.add_message("system", self.system)
            
    @property
    def supports_parallel_requests(self):
        """
        Returns True if the model supports parallel requests.
        """
        return strtobool(os.environ.get('INFERENCE_SUPPORTS_PARALLEL_REQUESTS', 'True'))

    @property
    def supports_frequency_penalty(self):
        """
        Returns True if the model supports frequency penalty.
        """
        return not 'claude' in self.model.lower() and not 'gemini' in self.model.lower()
    
    @property
    def supports_presence_penalty(self):
        """
        Returns True if the model supports presence penalty.
        """
        return not 'claude' in self.model.lower() and not 'gemini' in self.model.lower()
    
    @property
    def require_response_format(self):
        """
        Returns True if a response format is required.
        """
        return self.response_format is not None and issubclass(self.response_format, BaseModel)
    
    def add_message(self, role, content):
        """
        Adds a message to the conversation history.
        """
        content = self._sanitize_messages(content)
        if not isinstance(role, str) or role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        if self.require_response_format:
            content = content + f"Here is the JSON schema you need to follow for the response: {self.response_format.schema_json(indent=2)}\n"
            content += "Do not return the entire schema, only the response in JSON format. Use the schema only as a guide for filling in.\n"
        self.messages.append({"role": role, "content": content})

    def _sanitize_messages(self, content):
        # compact multiple spaces into one
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
        
    def get_request_params(self):
        """
        Returns the parameters for the LLM request.
        """
        params = {
            "model": self.model,
            "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
            # "top_p": self.top_p,
        }
        if self.supports_frequency_penalty:
            params['frequency_penalty'] = self.frequency_penalty
            
        if self.supports_presence_penalty:
            params['presence_penalty'] = self.presence_penalty
        
        # Add tools if enabled and available
        if self.enable_tools and self.tools:
            params['tools'] = [tool.to_openai_format() for tool in self.tools.values()]
            params['tool_choice'] = 'auto'  # Let the model decide when to use tools
            
        return params
    
    def get_embedding_params(self):
        """
        Returns the parameters for the embedding request.
        """
        params = {
            "model": self.model,
        }
        return params
    
    def embed(self, text):
        """
        Generate an embedding for the given text.
        """
        if not text:
            raise ValueError("Text cannot be empty for embedding.")
        
        params = self.get_embedding_params()
        logger.info(f"Embedding text with parameters: {params}")
        
        response = self._api_call_with_retry(
            self.client.embeddings.create,
            **params,
            input=text
        )
        
        return response.data[0].embedding if response.data else None
        
    def prompt(self, prompt=None, system=None, retry=False):
        """
        Generate a response from the LLM based on the provided prompt.
        """ 
        if prompt and not retry:
            # Reset retry count for new prompts
            self.retry_count = 0
            if system:
                self.add_message("system", system)
            self.add_message("user", prompt)
        
        # if self.require_response_format:
        #     self.think()
        
        params = self.get_request_params()
        logger.info(f"Prompting LLM with parameters: {params}")
        
        # Validate and sanitize messages before sending to LLM
        validated_messages = self._validate_messages_for_api(self.messages)
        
        response = self._api_call_with_retry(
            self.client.chat.completions.create,
            **params,
            messages=validated_messages
        )
        
        response = self.process_response(response)
        if not self.continuous:
            self.reset()
            
        return response
    
    def retry_prompt(self, prompt=None):
        """
        Retry the last prompt with the same parameters.
        """
        if self.retry_count >= self.max_retry:
            logger.warning(f"Maximum retry limit ({self.max_retry}) reached. Giving up on retry.")
            return None
        
        self.retry_count += 1
        logger.info(f"Retrying last prompt (attempt {self.retry_count}/{self.max_retry}).")
        return self.prompt(prompt, retry=True)

    def reset(self):
        """
        Reset the conversation history.
        """
        self.messages = []
        # Reset retry count when resetting conversation
        self.retry_count = 0
        if self.system:
            self.add_message("system", self.system)
    
    def _api_call_with_retry(self, api_func, *args, **kwargs):
        """
        Execute an API call with retry logic for transient errors.
        
        Retries on:
        - 403 Forbidden errors (often transient rate limiting)
        - 429 Rate limit errors
        - 500+ Server errors
        - Connection errors
        
        Args:
            api_func: The API function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The API response
            
        Raises:
            Exception: After max retries exceeded
        """
        max_retries = 3
        base_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"API call attempt {attempt + 1}/{max_retries}")
                response = api_func(*args, **kwargs)
                
                # Success - reset any retry tracking
                if attempt > 0:
                    logger.info(f"API call succeeded after {attempt + 1} attempts")
                    
                return response
                
            except APIStatusError as e:
                # Check for retryable status codes
                is_retryable = e.status_code in [403, 429, 500, 502, 503, 504]
                
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)  # Linear backoff: 5s, 10s, 15s
                    logger.warning(
                        f"API call failed with {e.status_code} error (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay} seconds... Error: {str(e)[:200]}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries exceeded
                    logger.error(
                        f"API call failed with {e.status_code} error after {attempt + 1} attempts. "
                        f"Error: {str(e)[:500]}"
                    )
                    raise
                    
            except (RateLimitError, APIConnectionError) as e:
                # These are always retryable
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    logger.warning(
                        f"API call failed with {type(e).__name__} (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay} seconds... Error: {str(e)[:200]}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"API call failed with {type(e).__name__} after {attempt + 1} attempts. "
                        f"Error: {str(e)[:500]}"
                    )
                    raise
                    
            except Exception as e:
                # Unexpected error - don't retry
                logger.error(f"API call failed with unexpected error: {type(e).__name__}: {str(e)[:500]}")
                raise
        
        # Should never reach here, but just in case
        raise Exception(f"API call failed after {max_retries} attempts")

    
    def add_tool(self, tool: ToolFunction):
        """
        Add a tool/function that the LLM can call.
        
        Args:
            tool: ToolFunction instance defining the tool
        """
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")
    
    def remove_tool(self, name: str):
        """
        Remove a tool by name.
        
        Args:
            name: Name of the tool to remove
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Removed tool: {name}")
        else:
            logger.warning(f"Tool not found: {name}")
    
    def add_tool_from_function(self, 
                               func: Callable, 
                               name: Optional[str] = None,
                               description: Optional[str] = None,
                               parameters: Optional[List] = None):
        """
        Add a tool from a Python function with automatic introspection.
        
        Args:
            func: Python function to wrap as a tool
            name: Optional name override (defaults to function name)
            description: Optional description override (defaults to docstring)
            parameters: Optional parameter definitions (will attempt to infer if not provided)
        """
        import inspect
        
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Function: {tool_name}"
        
        # Create basic tool function
        tool = ToolFunction(
            name=tool_name,
            description=tool_description,
            parameters=parameters or [],
            function=func
        )
        
        self.add_tool(tool)
    
    def list_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: ToolCall instance with name and arguments
            
        Returns:
            ToolResult with execution outcome
        """
        if tool_call.name not in self.tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' not found",
                success=False,
                error=f"Tool '{tool_call.name}' not found"
            )
        
        tool = self.tools[tool_call.name]
        if not tool.function:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' has no executable function",
                success=False,
                error="No executable function defined"
            )
        
        try:
            result = tool.function(**tool_call.arguments)
            
            # Validate and sanitize tool result content
            if result is None:
                content = "Tool executed successfully (no return value)"
                logger.warning(f"Tool {tool_call.name} returned None, using default content")
            elif str(result).strip() == "":
                content = "Tool executed successfully (empty result)"
                logger.warning(f"Tool {tool_call.name} returned empty result, using default content")
            elif str(result).lower() in ["none", "null"]:
                content = "Tool executed successfully (null result)"
                logger.warning(f"Tool {tool_call.name} returned null-like result, using default content")
            else:
                content = str(result).strip()
                # Ensure content is not just whitespace
                if not content:
                    content = "Tool executed successfully (whitespace result)"
                    logger.warning(f"Tool {tool_call.name} returned only whitespace, using default content")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                success=True
            )
        except Exception as e:
            logger.error(f"Tool execution error for {tool_call.name}: {str(e)}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing tool: {str(e)}",
                success=False,
                error=str(e)
            )
        
    def process_response(self, response):   
        """
        Process the response from the LLM, handling both regular messages and tool calls.
        """
        choice = response.choices[0]
        
        # Handle tool calls if present
        if (hasattr(choice.message, 'tool_calls') and 
            choice.message.tool_calls is not None):
            try:
                # Check if tool_calls is iterable and has items
                if hasattr(choice.message.tool_calls, '__iter__') and len(choice.message.tool_calls) > 0:
                    return self._handle_tool_calls(choice.message, response)
            except (TypeError, AttributeError):
                # If it's not iterable or doesn't have len(), it's likely a mock - skip tool handling
                pass
        
        # Handle regular message response
        if not choice.message.content:
            logger.warning("Received empty response from LLM")
            return ""
            
        message = choice.message.content.strip()
        
        # remove everything inside "<think> </think>" multiline tags
        message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)
        
        if '<response>' in message and '</response>' in message:
            message = re.search(r'<response>(.*?)</response>', message, flags=re.DOTALL).group(1)
        
        if len(message) > self.max_length:
            logger.warning(f"Max length exceeded: {len(message)} > {self.max_length}")
            retry_result = self.retry_prompt()
            if retry_result is not None:
                return retry_result
            # If retry failed or max retries reached, continue with current response
        
        self.add_message("assistant", message)
        
        if self.require_response_format:
            # remove all 'n/a' and 'none' from the message (case insensitive)
            
            logger.info(f"Original message: {message}")
            message = lm_json_utils.extract_strict_json(message)
            logger.info(f"Parsed message: {message}")
            return self.response_format.parse_raw(message)
        else:
            return response.choices[0].message.content
    
    def _handle_tool_calls(self, message, original_response):
        """
        Handle tool calls from the LLM response.
        
        Args:
            message: The message object containing tool calls
            original_response: The original response object
            
        Returns:
            The final response after executing tools
        """
        # Build the assistant's message with tool calls. Ensure content is never empty
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]
        }

        # Use the assistant content if provided and non-empty, otherwise provide a minimal valid content
        try:
            raw_content = message.content if hasattr(message, 'content') else None
        except Exception:
            raw_content = None

        if raw_content is not None and str(raw_content).strip():
            assistant_message["content"] = str(raw_content).strip()
        else:
            # OpenAI API requires non-empty content
            assistant_message["content"] = "[Tool calls in progress]"

        self.messages.append(assistant_message)
        
        # Execute each tool call
        tool_results = []
        for tool_call in message.tool_calls:
            try:
                # Parse arguments
                arguments = json.loads(tool_call.function.arguments)
                
                # Create ToolCall object
                tc = ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments
                )
                
                # Execute the tool
                result = self.execute_tool(tc)
                tool_results.append(result)
                
                # Add tool result to conversation with robust content validation
                tool_content = self._sanitize_tool_content(result.content, tool_call.function.name)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content
                })
                
                logger.info(f"Executed tool {tool_call.function.name}: {result.success}")
                
            except Exception as e:
                logger.error(f"Error processing tool call {tool_call.function.name}: {str(e)}")
                error_result = ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=f"Error: {str(e)}",
                    success=False,
                    error=str(e)
                )
                tool_results.append(error_result)
                
                # Add error to conversation with robust content validation
                error_content = self._sanitize_tool_content(error_result.content, tool_call.function.name, is_error=True)
                self.messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": error_content
                })
        
        # Get follow-up response from LLM after tool execution
        params = self.get_request_params()
        
        # Validate and sanitize messages before sending to LLM
        validated_messages = self._validate_messages_for_api(self.messages)
        
        # FINAL SAFETY CHECK before API call
        logger.info(f"🚀 Making follow-up API call with {len(validated_messages)} messages...")
        self._final_api_safety_check(validated_messages)
        
        # ABSOLUTE EMERGENCY CHECK - scan every message one last time
        for i, msg in enumerate(validated_messages):
            content = msg.get('content', '')
            if content == '':
                logger.error(f"🚨 CRITICAL: Found empty content at message {i} after ALL validation!")
                # Emergency fix
                if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                    msg['content'] = ' '  # Space for assistant with tool_calls
                else:
                    msg['content'] = '.'  # Period for others
                logger.info(f"🚨 EMERGENCY FIX: Set emergency content for message {i}")
        
        # Debug log specifically for message 5 if it exists
        if len(validated_messages) > 5:
            msg_5 = validated_messages[5]
            logger.info(f"🔍 DEBUG MESSAGE 5: role='{msg_5.get('role', 'MISSING')}', content='{msg_5.get('content', 'MISSING')}', has_tool_calls={'tool_calls' in msg_5}")
        
        follow_up_response = self._api_call_with_retry(
            self.client.chat.completions.create,
            **params,
            messages=validated_messages
        )
        
        # Process the follow-up response (this might include more tool calls)
        return self.process_response(follow_up_response)
    
    def enable_tool_support(self, enabled: bool = True):
        """
        Enable or disable tool support for this LLM instance.
        
        Args:
            enabled: Whether to enable tool support
        """
        self.enable_tools = enabled
        logger.info(f"Tool support {'enabled' if enabled else 'disabled'}")
    
    def clear_tools(self):
        """
        Remove all tools from this LLM instance.
        """
        self.tools.clear()
        logger.info("All tools cleared")
    
    def _validate_messages_for_api(self, messages):
        """
        Validate and sanitize messages before sending to LLM API.
        Ensures all messages have proper content and structure.
        AGGRESSIVE VALIDATION to prevent any API content errors.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of validated messages safe for API consumption
        """
        logger.info(f"🔍 VALIDATING {len(messages)} messages for API...")
        validated_messages = []
        
        for i, msg in enumerate(messages):
            logger.debug(f"Processing message {i}: {msg}")
            
            if not isinstance(msg, dict):
                logger.warning(f"❌ Skipping invalid message type at index {i}: {type(msg)}")
                continue
                
            # Create a deep copy to avoid modifying original
            validated_msg = {}
            for key, value in msg.items():
                validated_msg[key] = value
            
            # Ensure role is present first
            if 'role' not in validated_msg or not validated_msg['role']:
                logger.warning(f"❌ Skipping message at index {i} without role: {validated_msg}")
                continue
                
            # Ensure role is a valid string
            validated_msg['role'] = str(validated_msg['role']).strip()
            if not validated_msg['role']:
                logger.warning(f"❌ Skipping message at index {i} with empty role")
                continue
            
            # AGGRESSIVE CONTENT VALIDATION
            original_content = validated_msg.get('content')
            
            # Handle missing content field
            if 'content' not in validated_msg:
                logger.warning(f"⚠️ Message {i} missing content field")
                validated_msg['content'] = "Message content unavailable"
            
            # Handle None content
            elif validated_msg['content'] is None:
                logger.warning(f"⚠️ Message {i} has None content")
                validated_msg['content'] = "Message content unavailable"
            
            # Convert content to string and validate
            try:
                validated_msg['content'] = str(validated_msg['content'])
            except Exception as e:
                logger.error(f"❌ Failed to convert content to string for message {i}: {e}")
                validated_msg['content'] = "Message content conversion failed"
            
            # Apply role-specific content requirements
            role = validated_msg['role'].lower()
            content = validated_msg['content'].strip()
            
            if not content:  # Empty or whitespace-only content
                if role == 'tool':
                    validated_msg['content'] = "Tool executed successfully"
                    logger.info(f"✅ Set default tool content for message {i}")
                elif role == 'assistant' and 'tool_calls' in validated_msg:
                    # Assistant with tool_calls can have empty content
                    validated_msg['content'] = ""
                    logger.info(f"✅ Allowed empty content for assistant with tool_calls at message {i}")
                elif role in ['user', 'system']:
                    validated_msg['content'] = "Content unavailable"
                    logger.info(f"✅ Set default content for {role} message {i}")
                else:
                    validated_msg['content'] = "Content unavailable"
                    logger.info(f"✅ Set generic default content for {role} message {i}")
            else:
                # Content exists but might need role-specific handling
                if role == 'tool' and validated_msg['content'] in ["Message content unavailable", "Content unavailable"]:
                    validated_msg['content'] = "Tool executed successfully"
                    logger.info(f"✅ Fixed tool content for message {i}")
            
            # FINAL AGGRESSIVE VALIDATION - NO EMPTY CONTENT ALLOWED
            final_content = validated_msg['content']
            
            # The OpenAI API appears to be very strict about content
            # Ensure NO message has empty content, even assistants
            if final_content == "":
                if role == 'assistant' and 'tool_calls' in validated_msg:
                    # Even assistant with tool_calls should have some content for API safety
                    validated_msg['content'] = " "  # Single space (minimal valid content)
                    logger.info(f"✅ Set minimal space content for assistant with tool_calls at message {i}")
                else:
                    validated_msg['content'] = "."  # Minimal non-empty content
                    logger.info(f"✅ Set minimal content '.' for message {i}")
            
            # Double-check: ensure content length is never zero
            if len(validated_msg['content']) == 0:
                validated_msg['content'] = "."
                logger.warning(f"⚠️ Emergency fix: set minimal content for message {i}")
                
            # Triple-check: ensure content is never just whitespace if it would be empty
            if not validated_msg['content'].strip():
                if role == 'assistant' and 'tool_calls' in validated_msg:
                    validated_msg['content'] = " "  # Keep single space for tool_calls
                else:
                    validated_msg['content'] = "."  # Use period for others
                logger.warning(f"⚠️ Fixed whitespace-only content for message {i}")
            
            # Log the validation result
            logger.info(f"✅ Message {i} validated: role='{validated_msg['role']}', content_len={len(validated_msg['content'])}, content_preview='{validated_msg['content'][:50]}...'")
            
            validated_messages.append(validated_msg)
        
        # FINAL SAFETY CHECK - This should never find issues now
        logger.info(f"🔍 FINAL SAFETY CHECK on {len(validated_messages)} validated messages...")
        for i, msg in enumerate(validated_messages):
            issues = []
            
            if 'content' not in msg:
                issues.append("missing content field")
            elif msg['content'] is None:
                issues.append("None content")
            elif not isinstance(msg['content'], str):
                issues.append(f"content is {type(msg['content'])}, not string")
            elif len(msg['content']) == 0 and msg.get('role') != 'assistant':
                issues.append("empty content for non-assistant")
            elif len(msg['content']) == 0 and msg.get('role') == 'assistant' and 'tool_calls' not in msg:
                issues.append("empty content for assistant without tool_calls")
                
            if 'role' not in msg:
                issues.append("missing role field")
            elif not msg['role']:
                issues.append("empty role")
                
            if issues:
                logger.error(f"🚨 CRITICAL: Message {i} failed final validation: {issues} - {msg}")
                # Emergency fix
                if 'content' not in msg or msg['content'] is None or msg['content'] == "":
                    msg['content'] = "Emergency content fix"
                if 'role' not in msg or not msg['role']:
                    msg['role'] = "user"
                logger.error(f"🚨 Applied emergency fix to message {i}: {msg}")
        
        logger.info(f"✅ VALIDATION COMPLETE: {len(validated_messages)} messages ready for API")
        
        # Log the final message structure for debugging
        self._debug_log_messages(validated_messages)
        
        return validated_messages
    
    def _debug_log_messages(self, messages):
        """Debug log messages to help troubleshoot API issues."""
        logger.info(f"📋 DEBUG: Final message structure for API call:")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'MISSING_ROLE')
            content = msg.get('content', 'MISSING_CONTENT')
            content_preview = str(content)[:100] if content else 'EMPTY'
            has_tool_calls = 'tool_calls' in msg
            
            logger.info(f"  [{i}] role='{role}', content_len={len(str(content)) if content else 0}, has_tool_calls={has_tool_calls}")
            logger.info(f"      content_preview: '{content_preview}{'...' if len(str(content)) > 100 else ''}'")
            
            # Special validation for the problematic index
            if i == 5:  # The specific index from the error
                logger.info(f"🎯 SPECIAL ATTENTION - Message[5] (the problematic one):")
                logger.info(f"    Full message: {msg}")
                logger.info(f"    Content type: {type(content)}")
                logger.info(f"    Content repr: {repr(content)}")
                logger.info(f"    Content bool: {bool(content)}")
                if content == "":
                    logger.warning(f"⚠️ Message[5] has empty string content - this might cause the API error!")
                if content is None:
                    logger.error(f"🚨 Message[5] has None content - this WILL cause API error!")
                if 'content' not in msg:
                    logger.error(f"🚨 Message[5] missing 'content' field - this WILL cause API error!")
    
    def _final_api_safety_check(self, messages):
        """Final safety check before API call to prevent content errors."""
        logger.info("🔒 FINAL API SAFETY CHECK...")
        
        for i, msg in enumerate(messages):
            # Check for the exact conditions that cause the API error
            if not isinstance(msg, dict):
                logger.error(f"🚨 CRITICAL: Message {i} is not a dict: {type(msg)}")
                raise ValueError(f"Invalid message type at index {i}: {type(msg)}")
                
            if 'content' not in msg:
                logger.error(f"🚨 CRITICAL: Message {i} missing 'content' field")
                raise ValueError(f"Message {i} missing 'content' field")
                
            if msg['content'] is None:
                logger.error(f"🚨 CRITICAL: Message {i} has None content")
                raise ValueError(f"Message {i} has None content")
                
            if not isinstance(msg['content'], str):
                logger.error(f"🚨 CRITICAL: Message {i} content is not string: {type(msg['content'])}")
                raise ValueError(f"Message {i} content is not string: {type(msg['content'])}")
                
            # OpenAI API requires content for most message types
            if msg['content'] == "" and msg.get('role') != 'assistant':
                logger.error(f"🚨 CRITICAL: Message {i} has empty content for role {msg.get('role')}")
                # Emergency fix
                msg['content'] = "."
                logger.error(f"🚨 Applied emergency content fix to message {i}")
                
            if msg['content'] == "" and msg.get('role') == 'assistant' and 'tool_calls' not in msg:
                logger.error(f"🚨 CRITICAL: Message {i} assistant has empty content without tool_calls")
                # Emergency fix
                msg['content'] = "."
                logger.error(f"🚨 Applied emergency content fix to message {i}")
        
        logger.info("✅ Final safety check passed - all messages are API-safe")
    
    def _sanitize_tool_content(self, content, tool_name: str, is_error: bool = False) -> str:
        """
        Sanitize tool result content to ensure it's API-safe.
        
        Args:
            content: The raw tool result content
            tool_name: Name of the tool for logging
            is_error: Whether this is an error result
            
        Returns:
            API-safe content string
        """
        if content is None:
            default_msg = f"Error executing tool: {tool_name}" if is_error else f"Tool {tool_name} executed successfully (no return value)"
            logger.warning(f"Tool {tool_name} content is None, using default: {default_msg}")
            return default_msg
        
        # Convert to string and strip whitespace
        content_str = str(content).strip()
        
        if not content_str:
            default_msg = f"Error executing tool: {tool_name} (empty result)" if is_error else f"Tool {tool_name} executed successfully (empty result)"
            logger.warning(f"Tool {tool_name} returned empty content, using default: {default_msg}")
            return default_msg
        
        # Check for problematic string representations
        if content_str.lower() in ["none", "null", "undefined"]:
            default_msg = f"Error executing tool: {tool_name} (null result)" if is_error else f"Tool {tool_name} executed successfully (null result)"
            logger.warning(f"Tool {tool_name} returned null-like content '{content_str}', using default: {default_msg}")
            return default_msg
        
        # Ensure minimum length for API compatibility
        if len(content_str) < 1:
            default_msg = f"Error executing tool: {tool_name}" if is_error else f"Tool {tool_name} executed successfully"
            logger.warning(f"Tool {tool_name} content too short, using default: {default_msg}")
            return default_msg
        
        return content_str
