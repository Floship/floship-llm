from distutils.util import strtobool

from copy import copy
import logging
import re
from openai import OpenAI
import os
from pydantic import BaseModel, Field

from .utils import lm_json_utils # extract_and_fix_json, strict_json

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
        self.system = kwargs.get('system', None) # System prompt to set the context for the conversation
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
        logger.info("Retrying last prompt.")
        self.prompt(prompt, retry=True)

    def reset(self):
        """
        Reset the conversation history.
        """
        self.messages = []
        
    def process_response(self, response):   
        """
        Process the response from the LLM.
        """
        # try:
        message = response.choices[0].message.content.strip()
        
        # remove everything inside "<think> </think>" multiline tags
        message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)
        
        if '<response>' in message and '</response>' in message:
            message = re.search(r'<response>(.*?)</response>', message, flags=re.DOTALL).group(1)
        
        if len(message) > self.max_length:
            logger.warning(f"Max length exceeded: {len(message)} > {self.max_length}")
            self.retry_prompt()
        
        self.add_message("assistant", message)
        
        if self.require_response_format:
            # remove all 'n/a' and 'none' from the message (case insensitive)
            
            logger.info(f"Original message: {message}")
            message = lm_json_utils.extract_strict_json(message)
            logger.info(f"Parsed message: {message}")
            return self.response_format.parse_raw(message)
        else:
            return response.choices[0].message.content
        # except Exception as e:
            # logger.error(f"Error processing response: {e}")
            # return self.retry_prompt(f"Error processing response: {e}")
