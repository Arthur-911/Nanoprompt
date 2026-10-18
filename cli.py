"""Ultra-fast CLI for Prompt Token Optimization and ROI estimation."""
import sys, os, time, argparse, getpass
from core.tokenizer import TokenAnalyzer
from core.compressor import FastHeuristicCompressor, AICompressor, MAX_INPUT_LENGTH
from core.cost_estimator import CostEstimator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

console = Console()

def main():
    p = argparse.ArgumentParser(description="Optimize prompts for minimal token consumption.")
    p.add_argument("prompt", nargs="?", help="Human prompt to optimize")
    p.add_argument("--ai", action="store_true", help="Use deep AI compression")
    p.add_argument("--key", default=None, help="Gemini API Key")
    p.add_argument("--volume", type=int, default=100000, help="Projected request volume (default: 100,000)")
    args = p.parse_args()

    prompt = args.prompt or (console.print("[bold yellow]Enter prompt:[/bold yellow] ", end="") or input().strip())
    if not prompt: console.print("[bold red]Error: No prompt provided.[/bold red]"); sys.exit(1)
    if len(prompt) > MAX_INPUT_LENGTH: console.print(f"[bold red]Error: Input exceeds limit of {MAX_INPUT_LENGTH:,} characters.[/bold red]"); sys.exit(1)

    analyzer, start = TokenAnalyzer(), time.perf_counter()
    if args.ai:
        key = args.key
        if key: console.print("[bold yellow]Warning: Passing API keys via CLI flags exposes credentials. Use GEMINI_API_KEY env var.[/bold yellow]")
        elif not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            console.print("[cyan]Gemini API key not found in environment.[/cyan]")
            try: key = getpass.getpass("Enter Gemini API key (hidden, Enter to skip): ").strip() or None
            except Exception: pass
        comp = AICompressor(api_key=key, allow_env_fallback=True)
        if not comp.is_available():
            console.print("[yellow]Notice: Gemini key unavailable. Using fast offline engine.[/yellow]")
            optimized = FastHeuristicCompressor.compress(prompt)
        else:
            optimized, err = comp.compress(prompt)
            if err:
                console.print(f"[yellow]AI Notice: {err} Using offline fallback.[/yellow]")
                optimized = FastHeuristicCompressor.compress(prompt)
    else:
        try: optimized = FastHeuristicCompressor.compress(prompt)
        except ValueError as e: console.print(f"[bold red]Error: {e}[/bold red]"); sys.exit(1)

    duration_ms = (time.perf_counter() - start) * 1000
    metrics = analyzer.analyze_comparison(prompt, optimized)

    t = Table(title="Prompt Token Optimization Summary", show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="bold"); t.add_column("Original Input", style="red"); t.add_column("Optimized Prompt", style="green"); t.add_column("Difference / Savings", style="bold yellow")
    t.add_row("Token Count", str(metrics["original_tokens"]), str(metrics["optimized_tokens"]), f"-{metrics['tokens_saved']} tokens ({metrics['percentage_saved']}%)")
    t.add_row("Character Count", str(metrics["original_chars"]), str(metrics["optimized_chars"]), f"-{metrics['original_chars'] - metrics['optimized_chars']} chars ({metrics['char_reduction_percent']}%)")
    t.add_row("Execution Time", "-", "-", f"{duration_ms:.2f} ms")
    console.print(); console.print(t); console.print()

    roi = CostEstimator.estimate_savings(metrics["tokens_saved"], metrics["original_tokens"], metrics["optimized_tokens"], request_volume=args.volume)
    rt = Table(title=f"Projected Cost Savings & ROI (Volume: {args.volume:,} requests)", show_header=True, header_style="bold magenta")
    rt.add_column("Target Model", style="bold"); rt.add_column("Input Rate / 1M", justify="right"); rt.add_column(f"Original ({args.volume:,})", style="red", justify="right"); rt.add_column(f"Optimized ({args.volume:,})", style="green", justify="right"); rt.add_column("Net Savings", style="bold yellow", justify="right"); rt.add_column("Savings per 1M", style="bold cyan", justify="right")
    for name, info in roi["models"].items():
        rt.add_row(name, f"${info['rate_per_million']:.2f}", f"${info['original_cost']:,.2f}", f"${info['optimized_cost']:,.2f}", f"${info['savings']:,.2f}", f"${info['savings_per_1m']:,.2f}")
    console.print(rt); console.print()

    console.print(Panel(prompt, title="[bold red]Original Human Input[/bold red]", border_style="red"))
    console.print(Panel(optimized, title="[bold green]Optimized AI Prompt[/bold green]", border_style="green"))

if __name__ == "__main__": main()
