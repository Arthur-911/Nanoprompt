"""Tokenizer and Token Metrics module using tiktoken."""
from typing import Dict, Any
import tiktoken

MAX_INPUT_LENGTH = 10_000

class TokenAnalyzer:
    def __init__(self, model_encoding: str = "cl100k_base"):
        try:
            self.encoder = tiktoken.get_encoding(model_encoding)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text[:MAX_INPUT_LENGTH])) if text else 0

    def analyze_comparison(self, original_text: str, optimized_text: str) -> Dict[str, Any]:
        orig_t, opt_t = self.count_tokens(original_text), self.count_tokens(optimized_text)
        saved = orig_t - opt_t
        o_len, opt_len = len(original_text or ""), len(optimized_text or "")
        return {
            "original_tokens": orig_t,
            "optimized_tokens": opt_t,
            "tokens_saved": saved,
            "percentage_saved": round((saved / orig_t) * 100, 1) if orig_t > 0 else 0.0,
            "original_chars": o_len,
            "optimized_chars": opt_len,
            "char_reduction_percent": round(((o_len - opt_len) / max(o_len, 1)) * 100, 1) if o_len else 0.0
        }
