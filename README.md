# NanoPrompt: Prompt Token Optimizer and Analyzer

When interacting with Large Language Models (LLMs), people naturally use conversational habits: polite greetings, filler words, indirect requests, and pleasantries. While this feels natural, it unnecessarily increases token counts, raises API costs, and adds latency.

NanoPrompt is a lightweight utility designed to clean and streamline your prompts. It converts conversational requests into direct, imperative instructions while measuring exact token savings in real time.

---

## Overview

NanoPrompt provides two compression methods depending on your needs:

1. **Fast Heuristic Engine:** An instant, offline rule-based processor that removes polite hedges, redundant phrases, and conversational padding in under 5 milliseconds with zero API calls.
2. **Semantic AI Engine:** An optional deep restructuring engine powered by the Google GenAI SDK (Gemini) that reorganizes complex, multi-clause instructions into dense, structured prompts.

Exact token metrics are computed using Byte Pair Encoding (`tiktoken` cl100k_base), providing precise measurements of token reductions, character savings, and processing time.

---

## Key Capabilities

- **Instant Offline Processing:** Cleans common conversational padding (such as "Could you please help me to", "Thanks in advance", and "Basically I just want you to") without sending data over the network.
- **Precise Token Measurement:** Accurately compares original input against optimized output using the standard BPE tokenizer used by modern LLMs.
- **Dual Interfaces:** Use either the browser-based dashboard for interactive testing or the command-line interface for terminal workflows and scripts.
- **Built-in Security Safeguards:** Includes structural prompt injection protection (`<user_input>` boundary encapsulation), safe credential handling, input length ceilings to prevent memory exhaustion, and sanitized error reporting.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Arthur-911/Nanoprompt.git
   cd Nanoprompt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Optional: Set your Gemini API key if using the AI compression engine:
   ```bash
   # On Windows (PowerShell)
   $env:GEMINI_API_KEY="your-api-key-here"

   # On Linux / macOS
   export GEMINI_API_KEY="your-api-key-here"
   ```

---

## Usage

### 1. Interactive Web Dashboard

To launch the local web interface:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`. The interface provides:
- Live side-by-side prompt comparisons.
- Preset example queries for common tasks.
- Real-time token and character reduction metrics.
- Toggle between offline heuristic mode and AI restructuring mode.

### 2. Command Line Interface

For fast, single-command optimization directly in your terminal:

```bash
# Fast offline optimization
python cli.py "could you please help me do a pdf scan for me thanks in advance"

# AI-powered semantic optimization
python cli.py --ai "I was wondering if you could write some code to connect to PostgreSQL database and fetch users"
```

Example terminal output:

```text
                      Prompt Token Optimization Summary                       
+-----------------+----------------+------------------+----------------------+
| Metric          | Original Input | Optimized Prompt | Difference / Savings |
+-----------------+----------------+------------------+----------------------+
| Token Count     | 6              | 2                | -4 tokens (66.7%)    |
| Character Count | 20             | 8                | -12 chars (60.0%)    |
| Execution Time  | -              | -                | 3.47 ms              |
+-----------------+----------------+------------------+----------------------+

[Original Human Input]: do a pdf scan for me
[Optimized AI Prompt]:  Scan PDF
```

---

## Project Structure

```text
├── core/
│   ├── __init__.py       # Package exports for tokenizer and compressor classes
│   ├── compressor.py     # FastHeuristicCompressor and AICompressor implementations
│   └── tokenizer.py      # TokenAnalyzer class for BPE token counting and savings analysis
├── app.py                # Streamlit web dashboard
├── cli.py                # Terminal command-line interface with formatted output
├── test_compressor.py    # Automated benchmark and security validation test suite
└── requirements.txt      # Python dependencies with bounded versions
```

---

## Testing

Run the test suite to verify optimization benchmarks and security checks:

```bash
python test_compressor.py
```

---

## License

This project is open-source and available under the MIT License.
