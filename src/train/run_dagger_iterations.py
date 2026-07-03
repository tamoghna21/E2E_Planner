"""Phase 6a: run multiple DAgger iterations beyond the first.

Standard DAgger loop:
  1. Roll out current policy under real physics on training scenarios.
  2. Query the reactive tracking-controller expert at every visited state.
  3. Aggregate all prior data (D_bc ∪ all D_dagger so far) with the new batch.
  4. Retrain the same MLP/config on the aggregate.
  5. Evaluate the new policy on the held-out eval set (per dataset).
  Repeat.

Iteration 0 = BC-only baseline (bc_best.pt / metrics.json).
Iteration 1 = already done in Phase 4.6 (bc_dagger.pt / metrics_dagger.json).
This script runs iterations 2, 3, ... from there.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data.collect_dagger import aggregate_datasets, collect_dagger
from src.eval.closed_loop_eval import evaluate
from src.utils.metrics import aggregate_metrics_by_dataset, print_metrics_table
from src.train.train_bc import train

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


def run_iterations(start_iter, num_iters, start_ckpt, start_aggregate, max_dagger_steps=300):
    """Run DAgger iterations start_iter .. start_iter+num_iters-1."""
    current_ckpt = Path(start_ckpt)
    current_aggregate = Path(start_aggregate)
    results = []

    for iteration in range(start_iter, start_iter + num_iters):
        print(f"\n{'='*60}")
        print(f"DAgger iteration {iteration}")
        print(f"{'='*60}")
        print(f"  Policy: {current_ckpt}")
        print(f"  Aggregate data so far: {current_aggregate}")

        # Step 1+2: roll out current policy, query expert
        dagger_path = DATA_DIR / f"dagger_dataset_iter{iteration}.npz"
        dagger_obs, dagger_act, _ = collect_dagger(
            ckpt_path=str(current_ckpt), max_steps=max_dagger_steps
        )
        np.savez(dagger_path, obs=dagger_obs, act=dagger_act)
        print(f"\n  Collected {len(dagger_obs)} new transitions -> {dagger_path}")

        # Step 3: aggregate
        new_aggregate = DATA_DIR / f"aggregated_iter{iteration}.npz"
        aggregate_datasets(str(current_aggregate), str(dagger_path), str(new_aggregate))

        # Step 4: retrain
        ckpt_name = f"bc_dagger{iteration}.pt"
        loss_name = f"bc_dagger{iteration}_loss.png"
        train(data_path=str(new_aggregate), ckpt_name=ckpt_name, loss_plot_name=loss_name)
        new_ckpt = OUTPUTS_DIR / ckpt_name
        print(f"\n  Saved checkpoint: {new_ckpt}")

        # Step 5: evaluate
        metrics_path = OUTPUTS_DIR / f"metrics_dagger{iteration}.json"
        episodes = evaluate(ckpt_path=str(new_ckpt))
        metrics = aggregate_metrics_by_dataset(episodes)
        for dataset, m in metrics.items():
            print(f"\n  --- {dataset} ---")
            print_metrics_table(m)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved metrics: {metrics_path}")

        results.append({
            "iteration": iteration,
            "ckpt": str(new_ckpt),
            "metrics_path": str(metrics_path),
            "n_new_transitions": int(len(dagger_obs)),
            "overall_success_rate": metrics["overall"]["success_rate"],
            "overall_route_completion": metrics["overall"]["mean_route_completion"],
        })

        current_ckpt = new_ckpt
        current_aggregate = new_aggregate

    return results, current_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_iter", type=int, default=2)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--start_ckpt", type=str, default=str(OUTPUTS_DIR / "bc_dagger.pt"))
    parser.add_argument("--start_aggregate", type=str, default=str(DATA_DIR / "bc_dagger_dataset.npz"))
    parser.add_argument("--max_dagger_steps", type=int, default=300)
    args = parser.parse_args()

    results, final_ckpt = run_iterations(
        start_iter=args.start_iter,
        num_iters=args.num_iters,
        start_ckpt=args.start_ckpt,
        start_aggregate=args.start_aggregate,
        max_dagger_steps=args.max_dagger_steps,
    )
    print(f"\nFinal checkpoint: {final_ckpt}")
    print("\nIteration progression:")
    for r in results:
        print(f"  iter {r['iteration']}: success={r['overall_success_rate']*100:.1f}% "
              f"route_completion={r['overall_route_completion']*100:.1f}% "
              f"(+{r['n_new_transitions']} transitions)")
