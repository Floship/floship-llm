"""Provider backend abstractions for floship-llm."""

from floship_llm.backends.base import ProviderBackend
from floship_llm.backends.openai_compat import OpenAICompatibleBackend

__all__ = ["OpenAICompatibleBackend", "ProviderBackend"]

# NativeGeminiBackend is lazily imported to avoid requiring google-genai
# at import time.  Import from floship_llm.backends.native_gemini directly.
