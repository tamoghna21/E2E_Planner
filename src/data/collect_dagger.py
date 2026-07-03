"""Phase 4.6 step 3: diagnose and collect DAgger labels in one pass.

Roll out the current BC policy under real physics on the *training* scenarios (never eval). At
every visited state, also query the reactive tracking-controller expert (src/data/tracking_expert.py)
for the action *it* would take at that exact vehicle pose -- it's a closed-loop controller, so this
is well-defined off the logged path, unlike the old teleport-replay "expert". This yields two things
from one rollout:
  - the per-step action gap |expert_action - policy_action|, which diagnoses covariate shift
    directly (a large, growing steering gap near the road boundary means the expert would correct
    where BC doesn't -- compounding error, demonstrated rather than asserted);
  - the DAgger dataset itself: (visited_state, expert_action) pairs, the standard DAgger recipe of
    "drive with the policy, label with the expert".
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.tracking_expert import build_controller_for_scenario
from src.env.make_env import make_train_env, load_config
from src.eval.closed_loop_eval import load_policy, CKPT_PATH

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


def rollout_policy_query_expert(env, model, seed, max_steps, device="cpu"):
    """Drive with the BC policy; label each visited state with the reactive expert's action."""
    obs, info = env.reset(seed=seed)
    controller = build_controller_for_scenario(env)
    controller.reset()

    gaps, dagger_obs, dagger_act = [], [], []
    last_info = info
    for t in range(max_steps):
        with torch.no_grad():
            policy_action = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        policy_action = policy_action.squeeze(0).cpu().numpy()
        expert_action = controller.act(env.agent.position, env.agent.heading_theta, env.agent.speed)
        gaps.append(np.abs(expert_action - policy_action).tolist())
        dagger_obs.append(obs)
        dagger_act.append(expert_action.tolist())

        obs, r, terminated, truncated, info = env.step(policy_action)  # drive with the policy
        last_info = info
        if terminated or truncated:
            break

    return gaps, dagger_obs, dagger_act, last_info


def collect_dagger(ckpt_path=CKPT_PATH, config_path=None, out_dir=OUTPUTS_DIR, max_steps=300, device="cpu"):
    cfg = load_config(config_path) if config_path else load_config()
    model, _ = load_policy(ckpt_path, device)
    split = cfg["scenario_split"]
    train_indices = list(range(split["train_start"], split["train_start"] + split["train_num"]))

    env = make_train_env(config_path) if config_path else make_train_env()
    all_gaps = {}
    all_obs, all_act = [], []
    try:
        for idx in train_indices:
            gaps, dagger_obs, dagger_act, last_info = rollout_policy_query_expert(
                env, model, seed=idx, max_steps=max_steps, device=device
            )
            all_gaps[idx] = gaps
            all_obs.extend(dagger_obs)
            all_act.extend(dagger_act)
            mean_steering_gap = float(np.mean([g[0] for g in gaps])) if gaps else 0.0
            print(f"scenario {idx}: {len(gaps)} steps, mean_steering_gap={mean_steering_gap:.4f}, "
                  f"off_road={last_info.get('out_of_road', False)}, route_completion="
                  f"{last_info.get('route_completion', 0)*100:.1f}%")
    finally:
        env.close()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # action_gap.json: per-scenario series + summary
    summary = {}
    for idx, gaps in all_gaps.items():
        gaps_arr = np.asarray(gaps)
        summary[idx] = {
            "num_steps": len(gaps),
            "mean_steering_gap": float(gaps_arr[:, 0].mean()) if len(gaps) else 0.0,
            "mean_throttle_gap": float(gaps_arr[:, 1].mean()) if len(gaps) else 0.0,
            "final_steering_gap": float(gaps_arr[-1, 0]) if len(gaps) else 0.0,
            "steering_gap_series": gaps_arr[:, 0].tolist() if len(gaps) else [],
        }
    with open(out_dir / "action_gap.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots()
    for idx, row in summary.items():
        ax.plot(row["steering_gap_series"], label=f"scenario {idx}")
    ax.set_xlabel("rollout step (policy-driven)")
    ax.set_ylabel("|expert_steering - policy_steering|")
    ax.set_title("Action gap vs. step: policy-visited states, training scenarios")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "action_gap.png")
    plt.close(fig)
    print(f"Saved action gap diagnostics to {out_dir / 'action_gap.json'} / .png")

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    act_arr = np.asarray(all_act, dtype=np.float32)
    print(f"\nDAgger dataset collected: {obs_arr.shape[0]} new transitions "
          f"(states visited by the policy, labeled by the expert)")
    return obs_arr, act_arr, summary


def aggregate_datasets(bc_path, dagger_path, out_path):
    """D_aggregated = D_bc union D_dagger, saved as a single .npz for retraining."""
    bc = np.load(bc_path)
    dagger = np.load(dagger_path)
    obs = np.concatenate([bc["obs"], dagger["obs"]], axis=0).astype(np.float32)
    act = np.concatenate([bc["act"], dagger["act"]], axis=0).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs, act=act)
    print(f"D_bc ({len(bc['obs'])}) + D_dagger ({len(dagger['obs'])}) = {len(obs)} transitions "
          f"-> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "dagger_dataset.npz"))
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--aggregate_with", type=str, default=str(DATA_DIR / "bc_dataset.npz"),
                         help="path to D_bc; pass '' to skip producing the aggregated dataset")
    parser.add_argument("--aggregated_out", type=str, default=str(DATA_DIR / "bc_dagger_dataset.npz"))
    args = parser.parse_args()

    obs_arr, act_arr, summary = collect_dagger(ckpt_path=args.ckpt, max_steps=args.max_steps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_arr, act=act_arr)
    print(f"Saved DAgger dataset to {out_path}")

    if args.aggregate_with:
        aggregate_datasets(args.aggregate_with, out_path, args.aggregated_out)
