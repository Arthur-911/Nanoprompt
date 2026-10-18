"""Cost Estimator and ROI Analysis module for LLM token pricing."""
from typing import Dict, Any

DEFAULT_PRICING_PER_MILLION: Dict[str, float] = {
    "Claude 3.5 Sonnet": 3.00,
    "GPT-4o": 2.50,
    "Gemini 1.5 Pro": 1.25,
    "Claude 3.5 Haiku": 0.80,
    "GPT-4o mini": 0.15,
    "Gemini Flash": 0.075,
}

class CostEstimator:
    @staticmethod
    def calculate_cost(tokens: int, price_per_million: float) -> float:
        return (tokens / 1_000_000.0) * price_per_million if tokens > 0 and price_per_million > 0 else 0.0

    @classmethod
    def estimate_savings(
        cls, tokens_saved: int, original_tokens: int, optimized_tokens: int,
        request_volume: int = 100_000, pricing_table: Dict[str, float] = None
    ) -> Dict[str, Any]:
        pricing = pricing_table or DEFAULT_PRICING_PER_MILLION
        models = {}
        for name, rate in pricing.items():
            orig_c = cls.calculate_cost(original_tokens * request_volume, rate)
            opt_c = cls.calculate_cost(optimized_tokens * request_volume, rate)
            models[name] = {
                "rate_per_million": rate,
                "original_cost": round(orig_c, 2),
                "optimized_cost": round(opt_c, 2),
                "savings": round(orig_c - opt_c, 2),
                "savings_per_10k": round(cls.calculate_cost(tokens_saved * 10_000, rate), 2),
                "savings_per_100k": round(cls.calculate_cost(tokens_saved * 100_000, rate), 2),
                "savings_per_1m": round(cls.calculate_cost(tokens_saved * 1_000_000, rate), 2),
            }
        return {"tokens_saved_per_request": tokens_saved, "request_volume": request_volume, "models": models}
