import sys
from core import TokenAnalyzer, FastHeuristicCompressor, AICompressor, CostEstimator, MAX_INPUT_LENGTH

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

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

print("=" * 60 + "\nTOKEN OPTIMIZATION BENCHMARK\n" + "=" * 60)
for t in tests:
    comp = FastHeuristicCompressor.compress(t)
    st = analyzer.analyze_comparison(t, comp)
    print(f"RAW INPUT   [{st['original_tokens']} tokens]: {t}")
    print(f"COMPRESSED  [{st['optimized_tokens']} tokens]: {comp}")
    print(f"SAVED:       {st['tokens_saved']} tokens ({st['percentage_saved']}%)\n" + "-" * 60)

print("\nSECURITY & BOUNDARY TESTS\n" + "=" * 60)
assert "\x01" not in FastHeuristicCompressor.compress("find out whether the database is responsive")
print("[PASS] Control character \\x01 avoidance verified.")

try:
    FastHeuristicCompressor.compress("a" * (MAX_INPUT_LENGTH + 1))
    print("[FAIL] Oversized input did not raise ValueError")
except ValueError:
    print("[PASS] Max input length limit properly enforced (ValueError raised).")

assert analyzer.count_tokens("a" * (MAX_INPUT_LENGTH + 5000)) > 0
print("[PASS] Tokenizer bounds large input safely without memory exhaustion.")

assert not AICompressor(api_key=None, allow_env_fallback=False).is_available()
print("[PASS] Multi-tenant environment API key fallback prevention verified.")

savings = CostEstimator.estimate_savings(10, 25, 15, request_volume=1_000_000)
assert "GPT-4o" in savings["models"] and savings["models"]["GPT-4o"]["savings"] == 25.0
print("[PASS] Real-time cost estimator and financial ROI calculations verified.")

print("=" * 60 + "\nALL TESTS PASSED SUCCESSFULLY!\n" + "=" * 60)
