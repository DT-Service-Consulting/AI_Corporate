import argparse
import os
import subprocess
from datetime import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run split-robustness evaluation using dynamic stratified splits.")
    p.add_argument("--results", default="results.json")
    p.add_argument("--output-root", default="outputs/research_eval_split_robustness")
    p.add_argument("--split-seeds", default="101,202,303")
    p.add_argument("--router-seeds", default="42,43,44,45,46", help="Used when split-seeds is empty.")
    p.add_argument("--bootstrap-samples", type=int, default=4000)
    p.add_argument("--task-balance-power", type=float, default=1.5)
    p.add_argument("--uncertainty-threshold", type=float, default=0.7)
    p.add_argument("--dynamic-test-size", type=float, default=0.2)
    p.add_argument("--execute", action="store_true", help="Run commands. Without this flag, prints commands only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    split_seed_values = [s.strip() for s in args.split_seeds.split(",") if s.strip()]

    commands = []
    for split_seed in split_seed_values:
        out_dir = os.path.join(args.output_root, f"split_seed_{split_seed}")
        try:
            base_seed = int(split_seed)
            eval_seeds = ",".join(str(base_seed + i) for i in range(5))
        except ValueError:
            eval_seeds = args.router_seeds
        cmd = [
            "python",
            "evaluate_research_ready.py",
            "--results",
            args.results,
            "--output-dir",
            out_dir,
            "--seeds",
            eval_seeds,
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--task-balance-power",
            str(args.task_balance_power),
            "--uncertainty-threshold",
            str(args.uncertainty_threshold),
            "--dynamic-split",
            "--dynamic-test-size",
            str(args.dynamic_test_size),
        ]
        commands.append((split_seed, cmd))

    print("Planned split-robustness commands:")
    for split_seed, cmd in commands:
        print(f"[split_seed={split_seed}] {' '.join(cmd)}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to launch experiments.")
        return

    start = datetime.utcnow()
    for split_seed, cmd in commands:
        print(f"\nRunning split robustness for split_seed={split_seed}")
        subprocess.run(cmd, check=True)
    end = datetime.utcnow()
    print(f"Completed split-robustness suite. UTC start={start.isoformat()} end={end.isoformat()}")


if __name__ == "__main__":
    main()
