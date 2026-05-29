"""
CLI evaluation runner.

Usage:
    python run_evaluation.py --models oss frontier --output results/
    python run_evaluation.py --models frontier --skip-oss
    python run_evaluation.py --dry-run       # show prompts only, no inference
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Run AI assistant evaluation benchmark")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["oss", "frontier"],
        default=["oss", "frontier"],
        help="Which models to evaluate",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        choices=["factual", "adversarial", "bias"],
        default=["factual", "adversarial", "bias"],
        help="Which prompt categories to run",
    )
    p.add_argument("--output", default="results", help="Output directory")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without running inference",
    )
    p.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )
    return p.parse_args()


def _filter_prompts(prompts, categories):
    return [p for p in prompts if p["category"] in categories]


def _print_table(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'EVALUATION RESULTS':^70}")
    print("=" * 70)

    sections = {
        "Factual": [
            ("Accuracy", report["factual"]["oss"]["accuracy"], report["factual"]["frontier"]["accuracy"]),
            ("Completeness", report["factual"]["oss"]["completeness"], report["factual"]["frontier"]["completeness"]),
            ("Avg Latency (ms)", report["factual"]["oss"]["avg_latency_ms"], report["factual"]["frontier"]["avg_latency_ms"]),
        ],
        "Adversarial / Safety": [
            ("Safety Score", report["adversarial"]["oss"]["safety"], report["adversarial"]["frontier"]["safety"]),
            ("Jailbreak Robustness", report["adversarial"]["oss"]["robustness"], report["adversarial"]["frontier"]["robustness"]),
            ("Avg Latency (ms)", report["adversarial"]["oss"]["avg_latency_ms"], report["adversarial"]["frontier"]["avg_latency_ms"]),
        ],
        "Bias & Fairness": [
            ("Neutrality", report["bias"]["oss"]["neutrality"], report["bias"]["frontier"]["neutrality"]),
            ("Fairness", report["bias"]["oss"]["fairness"], report["bias"]["frontier"]["fairness"]),
            ("Avg Latency (ms)", report["bias"]["oss"]["avg_latency_ms"], report["bias"]["frontier"]["avg_latency_ms"]),
        ],
    }

    hdr = f"{'Metric':<28} {'OSS (Llama-3.1-8B)':>18} {'Frontier (Llama-3.3-70B)':>22}"
    for section, rows in sections.items():
        print(f"\n-- {section} --")
        print(hdr)
        print("-" * 66)
        for label, oss_val, frontier_val in rows:
            oss_str = f"{oss_val:.2f}" if isinstance(oss_val, float) else str(oss_val)
            fr_str = f"{frontier_val:.2f}" if isinstance(frontier_val, float) else str(frontier_val)
            # Latency: lower is better; all other metrics: higher is better
            is_latency = "Latency" in label
            if oss_val == frontier_val:
                winner = "Tie"
            elif is_latency:
                winner = "OSS slower  (!)" if oss_val > frontier_val else "OSS faster  (*)"
            else:
                winner = "Frontier >>" if frontier_val > oss_val else "<< OSS"
            print(f"  {label:<26} {oss_str:>18} {fr_str:>18}   {winner}")

    print("\n" + "=" * 70)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from evaluation.prompts import ALL_PROMPTS
    prompts = _filter_prompts(ALL_PROMPTS, args.categories)

    if args.dry_run:
        print(f"\n[dry-run] {len(prompts)} prompts across {args.categories}:")
        for p in prompts:
            print(f"  [{p['id']}] {p['category']}: {p['prompt'][:80]}…")
        return

    from evaluation.evaluator import Evaluator
    evaluator = Evaluator()

    oss_results = []
    frontier_results = []

    if "oss" in args.models:
        print("\n[eval] Running OSS assistant (Llama-3.1-8B-Instant via Groq)…")
        from assistants.oss_assistant import OSSAssistant
        oss_assistant = OSSAssistant()
        oss_results = evaluator.run_evaluation(oss_assistant, prompts, model_label="oss")
        oss_path = output_dir / "oss_results.json"
        oss_path.write_text(json.dumps(oss_results, indent=2, ensure_ascii=False))
        print(f"[eval] OSS results -> {oss_path}")

    if "frontier" in args.models:
        print("\n[eval] Running Frontier assistant (Llama-3.3-70B-Versatile via Groq)…")
        from assistants.frontier_assistant import FrontierAssistant
        frontier_assistant = FrontierAssistant()
        frontier_results = evaluator.run_evaluation(frontier_assistant, prompts, model_label="frontier")
        fr_path = output_dir / "frontier_results.json"
        fr_path.write_text(json.dumps(frontier_results, indent=2, ensure_ascii=False))
        print(f"[eval] Frontier results -> {fr_path}")

    # Load cached results if one model was skipped
    if "oss" not in args.models:
        cached = output_dir / "oss_results.json"
        if cached.exists():
            oss_results = json.loads(cached.read_text())
            print(f"[eval] Loaded cached OSS results from {cached}")

    if "frontier" not in args.models:
        cached = output_dir / "frontier_results.json"
        if cached.exists():
            frontier_results = json.loads(cached.read_text())
            print(f"[eval] Loaded cached Frontier results from {cached}")

    if oss_results and frontier_results:
        report = Evaluator.compare(oss_results, frontier_results)
        report_json = output_dir / "comparison_report.json"
        report_json.write_text(json.dumps(report, indent=2))
        print(f"[eval] Comparison report -> {report_json}")

        _print_table(report)

        if not args.no_charts:
            from evaluation.generate_report import generate_report, generate_latency_table
            from observability.logger import default_logger

            generate_report(report, output_path=output_dir / "evaluation_report.png")

            oss_metrics = default_logger.get_metrics("llama-3.1-8b")
            frontier_metrics = default_logger.get_metrics("groq-llama3.3-70b")
            generate_latency_table(
                oss_metrics,
                frontier_metrics,
                output_path=output_dir / "latency_cost_table.png",
            )
            print(f"\n[eval] Charts saved to: {output_dir}/")
    else:
        print("\n[eval] Need both OSS and Frontier results to generate comparison report.")


if __name__ == "__main__":
    main()
