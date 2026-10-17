# ⚡ Prompt Token Optimizer & Analyzer

A fast tool that converts natural, conversational human requests into token-minimized, high-impact AI prompts—and measures exact token savings in real time.

---

## 🚀 Key Features

- **⚡ Sub-Millisecond Compression Engine:** Cleans conversational filler, hedges, polite bloat, and passive constructs into direct, imperative prompts in `< 5ms` with zero API cost.
- **🤖 Deep AI Restructuring Mode (Optional):** Supports Gemini API for advanced semantic refactoring of large, complex prompts.
- **📊 Precise Token Metering:** Uses `tiktoken` (standard BPE tokenizer) to measure:
  - Original Human Input Tokens
  - Optimized AI Prompt Tokens
  - Exact Tokens Saved & Percentage Reduction
  - Character Reduction & Latency
- **🖥️ Dual Interfaces:**
  - **Interactive Web App (Streamlit):** Visual side-by-side comparison, preset examples, live metrics cards.
  - **Ultra-Fast CLI:** Instant command-line compression directly in your terminal.

---

## 🏃 Quick Start

### 1. Run the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run via Command Line (Instant Results)
```bash
python cli.py "do a pdf scan for me"
```

Output:
```
                      Prompt Token Optimization Summary                       
┌─────────────────┬────────────────┬──────────────────┬──────────────────────┐
│ Metric          │ Original Input │ Optimized Prompt │ Difference / Savings │
├─────────────────┼────────────────┼──────────────────┼──────────────────────┤
│ Token Count     │ 6              │ 2                │ -4 tokens (66.7%)    │
│ Character Count │ 20             │ 8                │ -12 chars (60.0%)    │
│ Execution Time  │ -              │ -                │ 3.73 ms              │
└─────────────────┴────────────────┴──────────────────┴──────────────────────┘

[Original Human Input]: do a pdf scan for me
[Optimized AI Prompt]:  Scan PDF
```

---

## 📁 Project Structure

```
├── core/
│   ├── __init__.py
│   ├── tokenizer.py      # BPE Tokenizer & savings metrics analyzer
│   └── compressor.py     # Fast heuristic engine + Gemini AI engine
├── app.py                # Streamlit Web UI
├── cli.py                # Fast terminal interface
└── test_compressor.py    # Benchmark test suite
```
