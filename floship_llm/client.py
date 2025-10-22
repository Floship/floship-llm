from distutils.util import strtobool

from copy import copy
import json
import logging
import re
from openai import OpenAI
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
        
        response = self.client.embeddings.create(
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
        response = self.client.chat.completions.create(
            **params,
            messages=self.messages
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
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=str(result),
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
        # Add the assistant's message with tool calls to conversation history
        self.messages.append({
            "role": "assistant",
            "content": message.content,
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
        })
        
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
                
                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content
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
                
                # Add error to conversation
                self.messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": error_result.content
                })
        
        # Get follow-up response from LLM after tool execution
        params = self.get_request_params()
        follow_up_response = self.client.chat.completions.create(
            **params,
            messages=self.messages
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
