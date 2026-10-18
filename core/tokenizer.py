"""
Tokenizer and Token Metrics module.
Uses tiktoken for precise BPE token counting and savings calculation.
"""
from typing import Dict, Any
import tiktoken

MAX_INPUT_LENGTH = 10_000

class TokenAnalyzer:
    def __init__(self, model_encoding: str = "cl100k_base"):
        """
        Initializes tokenizer.
        cl100k_base is standard for GPT-4, GPT-3.5-turbo, and provides
        a very accurate baseline for modern LLM token consumption.
        """
        try:
            self.encoder = tiktoken.get_encoding(model_encoding)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Returns the exact token count for the given text, safely bounded to MAX_INPUT_LENGTH."""
        if not text:
            return 0
        # Guard against unbounded memory allocation during BPE tokenization
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH]
        return len(self.encoder.encode(text))

    def analyze_comparison(self, original_text: str, optimized_text: str) -> Dict[str, Any]:
        """
        Computes detailed token metrics comparing original vs optimized text.
        """
        orig_tokens = self.count_tokens(original_text)
        opt_tokens = self.count_tokens(optimized_text)
        tokens_saved = orig_tokens - opt_tokens
        
        if orig_tokens > 0:
            percentage_saved = round((tokens_saved / orig_tokens) * 100, 1)
        else:
            percentage_saved = 0.0

        return {
            "original_tokens": orig_tokens,
            "optimized_tokens": opt_tokens,
            "tokens_saved": tokens_saved,
            "percentage_saved": percentage_saved,
            "original_chars": len(original_text),
            "optimized_chars": len(optimized_text),
            "char_reduction_percent": round(
                ((len(original_text) - len(optimized_text)) / max(len(original_text), 1)) * 100, 1
            ) if original_text else 0.0
        }
