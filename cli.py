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

from core.tokenizer import TokenAnalyzer
from core.compressor import FastHeuristicCompressor, AICompressor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Optimize prompts for minimal token consumption.")
    parser.add_argument("prompt", type=str, nargs="?", help="Human prompt to optimize")
    parser.add_argument("--ai", action="store_true", help="Use deep AI compression instead of fast offline rules")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key (optional)")

    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        console.print("[bold yellow]Enter your prompt:[/bold yellow] ", end="")
        prompt = input().strip()

    if not prompt:
        console.print("[bold red]Error: No prompt provided.[/bold red]")
        sys.exit(1)

    analyzer = TokenAnalyzer()
    start = time.perf_counter()

    if args.ai:
        compressor = AICompressor(api_key=args.key)
        if not compressor.is_available():
            console.print("[yellow]Notice: AI key not found, using fast offline engine.[/yellow]")
            optimized = FastHeuristicCompressor.compress(prompt)
        else:
            optimized, err = compressor.compress(prompt)
            if err:
                console.print(f"[red]AI error: {err}. Using offline fallback.[/red]")
                optimized = FastHeuristicCompressor.compress(prompt)
    else:
        optimized = FastHeuristicCompressor.compress(prompt)

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

    console.print(Panel(prompt, title="[bold red]Original Human Input[/bold red]", border_style="red"))
    console.print(Panel(optimized, title="[bold green]Optimized AI Prompt[/bold green]", border_style="green"))

if __name__ == "__main__":
    main()
