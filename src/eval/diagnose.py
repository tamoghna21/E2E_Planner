"""Phase 4.5 diagnostics: find out *why* BC fails closed-loop before naming a cause.

Three independent checks, run in order (see ROADMAP.md Phase 4.5):
  1. loss_by_dim: is the model underfitting one action dim (most likely throttle) while
     fitting the other fine?
  2. expert_replay_check (decisive): do our own pseudo-action labels, fed through real
     physics from the logged start state, actually reproduce the logged route? If not, BC
     was never going to work -- the problem is label/dynamics mismatch, not covariate shift.
  3. obs_drift (conditional on #2 vindicating the labels): does the policy's closed-loop
     observation distribution measurably drift from the training distribution over a rollout?
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metadrive.constants import TerminationState
from src.data.collect_expert import _pseudo_actions_for_track, _wheelbase_and_max_steering
from src.env.make_env import make_train_env, make_eval_env, make_eval_env_extra, load_config
from src.eval.closed_loop_eval import load_policy, run_episode
from src.train.train_bc import load_dataset, DATA_PATH

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
CKPT_PATH = OUTPUTS_DIR / "bc_best.pt"


# ---------------------------------------------------------------------------
# Step 1: per-dimension loss + naive baselines
# ---------------------------------------------------------------------------
def loss_by_dim(ckpt_path=CKPT_PATH, data_path=DATA_PATH, out_dir=OUTPUTS_DIR):
    cfg = load_config()
    model, _ = load_policy(ckpt_path, device="cpu")

    (train_obs, train_act), (val_obs, val_act) = load_dataset(data_path, cfg["train"]["val_fraction"])
    train_act, val_act = np.asarray(train_act), np.asarray(val_act)

    with torch.no_grad():
        pred = model(torch.as_tensor(val_obs, dtype=torch.float32)).numpy()

    train_mean = train_act.mean(axis=0)
    dim_names = ["steering", "throttle_brake"]
    table = {}
    for d, name in enumerate(dim_names):
        target = val_act[:, d]
        model_mse = float(np.mean((pred[:, d] - target) ** 2))
        zero_mse = float(np.mean((0.0 - target) ** 2))
        mean_mse = float(np.mean((train_mean[d] - target) ** 2))
        table[name] = {"model_mse": model_mse, "zero_baseline_mse": zero_mse, "train_mean_baseline_mse": mean_mse}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss_by_dim.json", "w") as f:
        json.dump(table, f, indent=2)

    fig, ax = plt.subplots()
    x = np.arange(len(dim_names))
    width = 0.25
    for i, key in enumerate(["model_mse", "zero_baseline_mse", "train_mean_baseline_mse"]):
        ax.bar(x + (i - 1) * width, [table[n][key] for n in dim_names], width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("val MSE")
    ax.set_title("Per-dimension model loss vs. naive baselines")
    ax.legend()
    fig.savefig(out_dir / "loss_by_dim.png")
    plt.close(fig)

    print("\n=== Per-dimension loss vs. baselines (val set) ===")
    for name in dim_names:
        row = table[name]
        print(f"{name:16s} model={row['model_mse']:.5f}  zero_baseline={row['zero_baseline_mse']:.5f}  "
              f"mean_baseline={row['train_mean_baseline_mse']:.5f}")
    return table


# ---------------------------------------------------------------------------
# Step 2 (decisive): replay our own pseudo-action labels through real physics
# ---------------------------------------------------------------------------
def expert_replay_check(config_path=None, out_dir=OUTPUTS_DIR, max_steps=1000):
    cfg = load_config(config_path) if config_path else load_config()
    split = cfg["scenario_split"]
    train_indices = list(range(split["train_start"], split["train_start"] + split["train_num"]))

    env = make_train_env(config_path) if config_path else make_train_env()
    results = []
    try:
        for idx in train_indices:
            obs, info = env.reset(seed=idx)
            scenario = env.engine.data_manager.current_scenario
            sdc_id = str(scenario["metadata"]["sdc_id"])
            track = scenario["tracks"][sdc_id]
            position, heading, speed, valid, curvature = _pseudo_actions_for_track(track)
            wheelbase, max_steering_rad = _wheelbase_and_max_steering(env.agent)

            n_transitions = min(len(position) - 1, max_steps)
            last_info = info
            steps_taken = 0
            for t in range(n_transitions):
                if not (valid[t] and valid[t + 1]):
                    continue
                steering = float(np.clip(np.arctan(wheelbase * curvature[t]) / max_steering_rad, -1.0, 1.0))
                accel = (speed[t + 1] - speed[t]) / 0.1
                throttle_brake = float(np.clip(accel / 3.0, -1.0, 1.0))
                obs, r, terminated, truncated, info = env.step([steering, throttle_brake])
                last_info = info
                steps_taken += 1
                if terminated or truncated:
                    break

            result = {
                "scenario_index": idx,
                "route_completion": float(last_info.get("route_completion", 0.0)),
                "success": bool(last_info.get(TerminationState.SUCCESS, False)),
                "collision": bool(last_info.get(TerminationState.CRASH, False)),
                "off_road": bool(last_info.get(TerminationState.OUT_OF_ROAD, False)),
                "num_steps": steps_taken,
            }
            results.append(result)
            print(f"scenario {idx}: route_completion={result['route_completion']*100:.1f}% "
                  f"success={result['success']} off_road={result['off_road']} collision={result['collision']} "
                  f"steps={steps_taken}")
    finally:
        env.close()

    mean_rc = float(np.mean([r["route_completion"] for r in results]))
    summary = {"mean_route_completion": mean_rc, "per_scenario": results}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "expert_replay_check.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmean route completion when replaying our own pseudo-action labels through physics: {mean_rc*100:.1f}%")
    return summary


# ---------------------------------------------------------------------------
# Step 3 (conditional): observation drift over a closed-loop rollout
# ---------------------------------------------------------------------------
def obs_drift(ckpt_path=CKPT_PATH, data_path=DATA_PATH, config_path=None, out_dir=OUTPUTS_DIR,
               seed=None, max_steps=300):
    from src.env.make_env import make_eval_env

    cfg = load_config(config_path) if config_path else load_config()
    if seed is None:
        seed = cfg["scenario_split"]["eval_start"]

    model, _ = load_policy(ckpt_path, device="cpu")
    train_obs = np.load(data_path)["obs"]

    env = make_eval_env(config_path) if config_path else make_eval_env()
    distances = []
    try:
        obs, info = env.reset(seed=seed)
        for _ in range(max_steps):
            d = float(np.min(np.linalg.norm(train_obs - obs[None, :], axis=1)))
            distances.append(d)
            with torch.no_grad():
                action = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            obs, r, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("nearest-neighbor L2 distance to training obs")
    ax.set_title(f"Observation drift over closed-loop rollout (scenario {seed})")
    fig.savefig(out_dir / "obs_drift.png")
    plt.close(fig)
    print(f"Saved obs drift plot to {out_dir / 'obs_drift.png'} ({len(distances)} steps)")
    return distances


# ---------------------------------------------------------------------------
# Phase 4.6 step 1: lateral-offset-vs-step drift, the low-dimensional signal the
# Phase 4.5 161-dim nearest-neighbor obs-drift test was insensitive to.
# ---------------------------------------------------------------------------
def lateral_drift(ckpt_path=CKPT_PATH, config_path=None, out_dir=OUTPUTS_DIR, max_steps=300, device="cpu"):
    cfg = load_config(config_path) if config_path else load_config()
    model, _ = load_policy(ckpt_path, device)
    split = cfg["scenario_split"]
    exclude = set(split.get("eval_exclude_indices", []))
    nuscenes_indices = [i for i in range(split["eval_start"], split["eval_start"] + split["eval_num"])
                        if i not in exclude]
    extra_num = cfg.get("eval_extra_num", 3)

    traces = {}
    env = make_eval_env(config_path) if config_path else make_eval_env()
    try:
        for idx in nuscenes_indices:
            _, deviations = run_episode(env, model, seed=idx, max_steps=max_steps, device=device,
                                         label=f"nuscenes:{idx}", return_trace=True)
            traces[f"nuscenes:{idx}"] = deviations
    finally:
        env.close()

    env_extra = make_eval_env_extra(config_path) if config_path else make_eval_env_extra()
    try:
        for idx in range(extra_num):
            _, deviations = run_episode(env_extra, model, seed=idx, max_steps=max_steps, device=device,
                                         label=f"waymo:{idx}", return_trace=True)
            traces[f"waymo:{idx}"] = deviations
    finally:
        env_extra.close()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for label, deviations in traces.items():
        ax.plot(deviations, label=label)
    ax.axhline(4.0, color="red", linestyle="--", label="off-road threshold (max_lateral_dist=4m)")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("abs(lateral offset from logged route) [m]")
    ax.set_title("Closed-loop lateral drift vs. step (BC policy, all eval scenarios)")
    ax.legend(fontsize=7)
    fig.savefig(out_dir / "lateral_drift.png")
    plt.close(fig)
    print(f"Saved lateral drift plot to {out_dir / 'lateral_drift.png'}")
    return traces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["loss_by_dim", "expert_replay_check", "obs_drift", "lateral_drift", "all"],
                         default="all")
    args = parser.parse_args()

    if args.step in ("loss_by_dim", "all"):
        loss_by_dim()
    if args.step in ("expert_replay_check", "all"):
        expert_replay_check()
    if args.step in ("lateral_drift", "all"):
        lateral_drift()
    if args.step == "obs_drift":
        obs_drift()
