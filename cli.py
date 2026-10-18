"""
Ultra-fast Command Line Interface for Prompt Token Optimization.
Usage:
    python cli.py "do a pdf scan for me"
    python cli.py --ai "Hey could you please analyze this..."
"""
import sys
import argparse
import time

# Ensure safe UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import getpass
from core.tokenizer import TokenAnalyzer
from core.compressor import FastHeuristicCompressor, AICompressor, MAX_INPUT_LENGTH
from core.cost_estimator import CostEstimator, DEFAULT_PRICING_PER_MILLION
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Optimize prompts for minimal token consumption.")
    parser.add_argument("prompt", type=str, nargs="?", help="Human prompt to optimize")
    parser.add_argument("--ai", action="store_true", help="Use deep AI compression instead of fast offline rules")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key (Passing via CLI argument is discouraged; prefer GEMINI_API_KEY env var)")
    parser.add_argument("--volume", type=int, default=100000, help="Projected request volume for ROI calculations (default: 100,000)")

    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        console.print("[bold yellow]Enter your prompt:[/bold yellow] ", end="")
        prompt = input().strip()

    if not prompt:
        console.print("[bold red]Error: No prompt provided.[/bold red]")
        sys.exit(1)

    if len(prompt) > MAX_INPUT_LENGTH:
        console.print(f"[bold red]Error: Input length ({len(prompt):,} characters) exceeds maximum supported limit of {MAX_INPUT_LENGTH:,} characters.[/bold red]")
        sys.exit(1)

    analyzer = TokenAnalyzer()
    start = time.perf_counter()

    if args.ai:
        api_key = args.key
        if api_key:
            console.print("[bold yellow]Security Warning: Passing API keys via CLI arguments can expose credentials in process lists and shell logs. Prefer setting GEMINI_API_KEY environment variable.[/bold yellow]")
        elif not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            console.print("[cyan]Gemini API key not found in environment.[/cyan]")
            try:
                entered_key = getpass.getpass("Enter Gemini API key (input hidden, or Enter to skip): ").strip()
                if entered_key:
                    api_key = entered_key
            except Exception:
                pass

        compressor = AICompressor(api_key=api_key, allow_env_fallback=True)
        if not compressor.is_available():
            console.print("[yellow]Notice: Gemini API key unavailable. Using fast offline engine.[/yellow]")
            optimized = FastHeuristicCompressor.compress(prompt)
        else:
            optimized, err = compressor.compress(prompt)
            if err:
                console.print(f"[yellow]AI Notice: {err} Using offline fallback.[/yellow]")
                optimized = FastHeuristicCompressor.compress(prompt)
    else:
        try:
            optimized = FastHeuristicCompressor.compress(prompt)
        except ValueError as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            sys.exit(1)

    duration_ms = (time.perf_counter() - start) * 1000
    metrics = analyzer.analyze_comparison(prompt, optimized)

    # Output Table
    table = Table(title="Prompt Token Optimization Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Original Input", style="red")
    table.add_column("Optimized Prompt", style="green")
    table.add_column("Difference / Savings", style="bold yellow")

    table.add_row(
        "Token Count",
        str(metrics["original_tokens"]),
        str(metrics["optimized_tokens"]),
        f"-{metrics['tokens_saved']} tokens ({metrics['percentage_saved']}%)"
    )
    table.add_row(
        "Character Count",
        str(metrics["original_chars"]),
        str(metrics["optimized_chars"]),
        f"-{metrics['original_chars'] - metrics['optimized_chars']} chars ({metrics['char_reduction_percent']}%)"
    )
    table.add_row(
        "Execution Time",
        "-",
        "-",
        f"{duration_ms:.2f} ms"
    )

    console.print()
    console.print(table)
    console.print()

    # Cost Savings and ROI Table
    roi_data = CostEstimator.estimate_savings(
        tokens_saved=metrics["tokens_saved"],
        original_tokens=metrics["original_tokens"],
        optimized_tokens=metrics["optimized_tokens"],
        request_volume=args.volume
    )

    roi_table = Table(
        title=f"Projected Cost Savings & ROI (Volume: {args.volume:,} requests)",
        show_header=True,
        header_style="bold magenta"
    )
    roi_table.add_column("Target Model", style="bold")
    roi_table.add_column("Input Rate / 1M", justify="right")
    roi_table.add_column(f"Original ({args.volume:,})", style="red", justify="right")
    roi_table.add_column(f"Optimized ({args.volume:,})", style="green", justify="right")
    roi_table.add_column("Net Savings", style="bold yellow", justify="right")
    roi_table.add_column("Savings per 1M", style="bold cyan", justify="right")

    for model_name, info in roi_data["models"].items():
        roi_table.add_row(
            model_name,
            f"${info['rate_per_million']:.2f}",
            f"${info['original_cost']:,.2f}",
            f"${info['optimized_cost']:,.2f}",
            f"${info['savings']:,.2f}",
            f"${info['savings_per_1m']:,.2f}"
        )

    console.print(roi_table)
    console.print()

    console.print(Panel(prompt, title="[bold red]Original Human Input[/bold red]", border_style="red"))
    console.print(Panel(optimized, title="[bold green]Optimized AI Prompt[/bold green]", border_style="green"))

if __name__ == "__main__":
    main()
