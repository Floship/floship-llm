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
                response = api_func(*args, **kwargs)
                
                # Success - log only if retry occurred
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
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of validated messages safe for API consumption
        """
        validated_messages = []
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(f"Skipping invalid message type at index {i}: {type(msg)}")
                continue
                
            # Create a deep copy to avoid modifying original
            validated_msg = {}
            for key, value in msg.items():
                validated_msg[key] = value
            
            # Ensure role is present and valid
            if 'role' not in validated_msg or not validated_msg['role']:
                logger.warning(f"Skipping message at index {i} without role")
                continue
                
            validated_msg['role'] = str(validated_msg['role']).strip()
            if not validated_msg['role']:
                logger.warning(f"Skipping message at index {i} with empty role")
                continue
            
            # Handle content validation
            if 'content' not in validated_msg:
                validated_msg['content'] = "Message content unavailable"
            elif validated_msg['content'] is None:
                validated_msg['content'] = "Message content unavailable"
            
            # Convert content to string
            try:
                validated_msg['content'] = str(validated_msg['content'])
            except Exception as e:
                logger.error(f"Failed to convert content to string for message {i}: {e}")
                validated_msg['content'] = "Message content conversion failed"
            
            # Apply role-specific content requirements
            role = validated_msg['role'].lower()
            content = validated_msg['content'].strip()
            
            if not content:
                if role == 'tool':
                    validated_msg['content'] = "Tool executed successfully"
                elif role == 'assistant' and 'tool_calls' in validated_msg:
                    validated_msg['content'] = " "  # Minimal space for API compatibility
                elif role in ['user', 'system']:
                    validated_msg['content'] = "Content unavailable"
                else:
                    validated_msg['content'] = "Content unavailable"
            else:
                # Fix tool content if needed
                if role == 'tool' and validated_msg['content'] in ["Message content unavailable", "Content unavailable"]:
                    validated_msg['content'] = "Tool executed successfully"
            
            # Final validation - ensure no empty content except for assistant with tool_calls
            if validated_msg['content'] == "":
                if role == 'assistant' and 'tool_calls' in validated_msg:
                    validated_msg['content'] = " "
                else:
                    validated_msg['content'] = "."
            
            # Ensure content is never just whitespace
            if not validated_msg['content'].strip():
                if role == 'assistant' and 'tool_calls' in validated_msg:
                    validated_msg['content'] = " "
                else:
                    validated_msg['content'] = "."
            
            validated_messages.append(validated_msg)
        
        # Final safety check
        for i, msg in enumerate(validated_messages):
            if 'content' not in msg or msg['content'] is None or (msg['content'] == "" and msg.get('role') != 'assistant'):
                logger.error(f"Critical validation error at message {i}")
                if 'content' not in msg or msg['content'] is None or msg['content'] == "":
                    msg['content'] = "Emergency content fix"
                if 'role' not in msg or not msg['role']:
                    msg['role'] = "user"
        
        return validated_messages
    
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
