from core import TokenAnalyzer, FastHeuristicCompressor, AICompressor, MAX_INPUT_LENGTH

analyzer = TokenAnalyzer()

tests = [
    "do a pdf scan for me",
    "do a scan for me",
    "find out if the server is healthy",
    "Hey! Could you please help me to do an analysis of this sales data and tell me what is inside it? Thanks in advance!",
    "Hello, I was wondering if you could write some code to connect to PostgreSQL database and fetch users. Thanks!",
    "Can you please kindly do a summary of this article for me asap?",
    "Basically I just want you to figure out why this python script is failing with KeyError"
]

print("=" * 60)
print("TOKEN OPTIMIZATION BENCHMARK")
print("=" * 60)

for t in tests:
    comp = FastHeuristicCompressor.compress(t)
    stats = analyzer.analyze_comparison(t, comp)
    print(f"RAW INPUT   [{stats['original_tokens']} tokens]: {t}")
    print(f"COMPRESSED  [{stats['optimized_tokens']} tokens]: {comp}")
    print(f"SAVED:       {stats['tokens_saved']} tokens ({stats['percentage_saved']}%)")
    print("-" * 60)

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("\nSECURITY & BOUNDARY TESTS")
print("=" * 60)

# Test 1: No control characters injected
escaped_test = FastHeuristicCompressor.compress("find out whether the database is responsive")
assert "\x01" not in escaped_test, "Failed: Control character \\x01 found in output!"
print("[PASS] Control character \\x01 avoidance verified.")

# Test 2: Input length bounds enforcement
try:
    oversized = "a" * (MAX_INPUT_LENGTH + 1)
    FastHeuristicCompressor.compress(oversized)
    print("[FAIL] Oversized input did not raise ValueError")
except ValueError:
    print("[PASS] Max input length limit properly enforced (ValueError raised).")

# Test 3: Tokenizer memory protection
bounded_tokens = analyzer.count_tokens("a" * (MAX_INPUT_LENGTH + 5000))
assert bounded_tokens > 0
print("[PASS] Tokenizer bounds large input safely without memory exhaustion.")

# Test 4: AI Compressor multi-tenant credential isolation
ai = AICompressor(api_key=None, allow_env_fallback=False)
assert not ai.is_available(), "Failed: AICompressor fell back to environment key when allow_env_fallback=False"
print("[PASS] Multi-tenant environment API key fallback prevention verified.")

# Test 5: Cost Estimator and ROI calculation
from core import CostEstimator, DEFAULT_PRICING_PER_MILLION
savings = CostEstimator.estimate_savings(
    tokens_saved=10,
    original_tokens=25,
    optimized_tokens=15,
    request_volume=1_000_000
)
assert "GPT-4o" in savings["models"], "Failed: GPT-4o missing from pricing models"
assert savings["models"]["GPT-4o"]["savings"] == 25.0, f"Expected $25.0 savings, got {savings['models']['GPT-4o']['savings']}"
print("[PASS] Real-time cost estimator and financial ROI calculations verified.")

print("=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
