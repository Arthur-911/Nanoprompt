from core import TokenAnalyzer, FastHeuristicCompressor

analyzer = TokenAnalyzer()

tests = [
    "do a pdf scan for me",
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
