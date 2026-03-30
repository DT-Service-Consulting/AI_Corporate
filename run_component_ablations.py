import argparse
import os
import subprocess


ABLATIONS = [
    ("full_on", "on", "on"),
    ("no_chunking", "off", "on"),
    ("no_not_found_norm", "on", "off"),
    ("no_chunking_no_not_found_norm", "off", "off"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run component ablations for chunking and NOT_FOUND normalization.")
    p.add_argument("--output-root", default="outputs/component_ablations")
    p.add_argument("--results-prefix", default="results_ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rate-limit-s", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-backoff-s", type=float, default=1.5)
    p.add_argument("--eval-seeds", default="42,43,44,45,46")
    p.add_argument("--bootstrap-samples", type=int, default=4000)
    p.add_argument("--task-balance-power", type=float, default=1.5)
    p.add_argument("--uncertainty-threshold", type=float, default=0.7)
    p.add_argument("--execute", action="store_true", help="Run commands. Without this flag, prints commands only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    commands = []
    for name, chunking, not_found_norm in ABLATIONS:
        results_path = f"{args.results_prefix}_{name}.json"
        eval_out = os.path.join(args.output_root, name)
        run_cmd = [
            "python",
            "run_experiment.py",
            "--output",
            results_path,
            "--seed",
            str(args.seed),
            "--rate-limit-s",
            str(args.rate_limit_s),
            "--max-retries",
            str(args.max_retries),
            "--retry-backoff-s",
            str(args.retry_backoff_s),
            "--extraction-chunking",
            chunking,
            "--canonical-not-found",
            not_found_norm,
        ]
        eval_cmd = [
            "python",
            "evaluate_research_ready.py",
            "--results",
            results_path,
            "--output-dir",
            eval_out,
            "--seeds",
            args.eval_seeds,
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--task-balance-power",
            str(args.task_balance_power),
            "--uncertainty-threshold",
            str(args.uncertainty_threshold),
        ]
        commands.append((name, run_cmd, eval_cmd))

    print("Planned component ablation commands:")
    for name, run_cmd, eval_cmd in commands:
        print(f"\n[{name}] run:  {' '.join(run_cmd)}")
        print(f"[{name}] eval: {' '.join(eval_cmd)}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to launch ablation experiments.")
        return

    for name, run_cmd, eval_cmd in commands:
        print(f"\nRunning ablation: {name}")
        subprocess.run(run_cmd, check=True)
        subprocess.run(eval_cmd, check=True)
    print("\nCompleted component ablation suite.")


if __name__ == "__main__":
    main()
