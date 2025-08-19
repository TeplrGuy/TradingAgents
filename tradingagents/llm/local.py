"""Local LLM adapters for TradingAgents.

This module provides abstractions for integrating various local LLM providers
and models, supporting both API-based and direct inference approaches.
"""

import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Try to import langchain dependencies, but make them optional for testing
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.outputs import ChatGeneration, ChatResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Define minimal mock classes for testing without langchain
    LANGCHAIN_AVAILABLE = False
    
    class BaseMessage:
        def __init__(self, content: str):
            self.content = content
    
    class HumanMessage(BaseMessage):
        pass
    
    class AIMessage(BaseMessage):
        pass
    
    class SystemMessage(BaseMessage):
        pass
    
    class BaseChatModel:
        def __init__(self, **kwargs):
            pass
    
    class CallbackManagerForLLMRun:
        pass
    
    class ChatGeneration:
        def __init__(self, message, generation_info=None):
            self.message = message
            self.generation_info = generation_info or {}
    
    class ChatResult:
        def __init__(self, generations):
            self.generations = generations


class LocalLLMAdapter(BaseChatModel, ABC):
    """Base adapter class for local LLM providers.
    
    This abstract class provides a common interface for all local LLM implementations,
    ensuring compatibility with the existing TradingAgents architecture.
    """
    
    def __init__(self, model_name: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
    
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the local LLM."""
        pass
    
    @property
    def _llm_type(self) -> str:
        return "local_llm"
    
    def _format_messages_for_api(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI-compatible format."""
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # Default fallback
            
            formatted_messages.append({
                "role": role,
                "content": message.content
            })
        return formatted_messages


class APIBasedLocalLLM(LocalLLMAdapter):
    """Adapter for API-based local LLM providers (LM Studio, Text Generation WebUI, etc.).
    
    This adapter communicates with local LLM servers via REST API calls,
    supporting OpenAI-compatible endpoints.
    """
    
    def __init__(
        self,
        base_url: str,
        model_name: str = "local-model",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize the API-based local LLM adapter.
        
        Args:
            base_url: Base URL of the local LLM server
            model_name: Name of the model to use
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Ensure base_url ends with proper endpoint
        if not self.base_url.endswith('/chat/completions'):
            if self.base_url.endswith('/v1'):
                self.base_url += '/chat/completions'
            else:
                self.base_url += '/v1/chat/completions'
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using API call to local LLM server."""
        
        # Format messages for API
        formatted_messages = self._format_messages_for_api(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }
        
        # Add optional parameters
        if self.max_tokens:
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        if stop:
            payload["stop"] = stop
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Make API call
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response format from local LLM API")
            
            content = response_data["choices"][0]["message"]["content"]
            
            # Create ChatGeneration
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info=response_data.get("usage", {})
            )
            
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with local LLM API at {self.base_url}: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid response from local LLM API: {e}")
    
    def bind_tools(self, tools):
        """Bind tools to the model (basic implementation for compatibility)."""
        # For local models, tool binding might not be supported
        # Return self to maintain interface compatibility
        return self


class DirectInferenceLocalLLM(LocalLLMAdapter):
    """Adapter for direct Python inference with local LLMs.
    
    This adapter loads and runs local LLM models directly in Python,
    using libraries like transformers, llama-cpp-python, etc.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "transformers",
        device: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize the direct inference local LLM adapter.
        
        Args:
            model_path: Path to the local model files
            model_type: Type of model loader ("transformers", "llama_cpp", etc.)
            device: Device to run the model on ("cpu", "cuda", "auto")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(model_name=model_path, **kwargs)
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the local model based on the specified type."""
        
        if self.model_type == "transformers":
            self._load_transformers_model()
        elif self.model_type == "llama_cpp":
            self._load_llama_cpp_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_transformers_model(self):
        """Load model using Hugging Face transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("transformers and torch are required for transformers model type")
    
    def _load_llama_cpp_model(self):
        """Load model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context length
                n_gpu_layers=-1 if self.device != "cpu" else 0
            )
            
        except ImportError:
            raise ImportError("llama-cpp-python is required for llama_cpp model type")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using direct model inference."""
        
        if self.model_type == "transformers":
            return self._generate_transformers(messages, stop, **kwargs)
        elif self.model_type == "llama_cpp":
            return self._generate_llama_cpp(messages, stop, **kwargs)
        else:
            raise ValueError(f"Generation not implemented for model type: {self.model_type}")
    
    def _generate_transformers(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate using transformers model."""
        import torch
        
        # Convert messages to text prompt
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Create ChatGeneration
        generation = ChatGeneration(
            message=AIMessage(content=generated_text.strip()),
            generation_info={}
        )
        
        return ChatResult(generations=[generation])
    
    def _generate_llama_cpp(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate using llama-cpp model."""
        
        # Convert messages to text prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate
        output = self.model(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stop=stop or [],
        )
        
        content = output["choices"][0]["text"]
        
        # Create ChatGeneration
        generation = ChatGeneration(
            message=AIMessage(content=content.strip()),
            generation_info=output.get("usage", {})
        )
        
        return ChatResult(generations=[generation])
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def bind_tools(self, tools):
        """Bind tools to the model (basic implementation for compatibility)."""
        # For local models, tool binding might not be supported
        # Return self to maintain interface compatibility
        return self


def create_local_llm(
    provider: str,
    model_name: str,
    base_url: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> LocalLLMAdapter:
    """Factory function to create appropriate local LLM adapter.
    
    Args:
        provider: Local LLM provider type ("lm_studio", "text_gen_webui", "custom_api", "direct")
        model_name: Name or path of the model
        base_url: Base URL for API-based providers
        model_path: Path to model files for direct inference
        **kwargs: Additional configuration options
    
    Returns:
        Configured LocalLLMAdapter instance
    """
    
    # Normalize provider name
    provider_normalized = provider.lower().replace(" ", "_").replace("local_api", "api")
    
    api_providers = ["lm_studio", "text_generation_webui", "custom_api", "ollama"]
    
    if provider_normalized in api_providers:
        if not base_url:
            raise ValueError(f"base_url is required for provider: {provider}")
        
        return APIBasedLocalLLM(
            base_url=base_url,
            model_name=model_name,
            **kwargs
        )
    
    elif provider_normalized in ["direct_inference", "direct"]:
        if not model_path:
            raise ValueError("model_path is required for direct inference")
        
        return DirectInferenceLocalLLM(
            model_path=model_path,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported local LLM provider: {provider}")