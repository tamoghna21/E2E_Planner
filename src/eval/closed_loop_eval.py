"""The real result: let the trained BC policy actually drive the ego through held-out scenarios
while logged agents replay around it, and measure closed-loop driving quality (not loss).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metadrive.constants import TerminationState
from src.env.make_env import make_eval_env, make_eval_env_extra, load_config
from src.models.mlp_policy import MLPPolicy
from src.utils.metrics import EpisodeResult, aggregate_metrics, aggregate_metrics_by_dataset, print_metrics_table

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "outputs" / "bc_best.pt"
OUTPUTS_DIR = ROOT / "outputs"


def load_policy(ckpt_path=CKPT_PATH, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]["model"]
    model = MLPPolicy(obs_dim=161, act_dim=2, hidden_sizes=tuple(model_cfg["hidden_sizes"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["config"]


def run_episode(env, model, seed, max_steps, device="cpu", label=None, return_trace=False):
    obs, info = env.reset(seed=seed)
    deviations = []
    last_info = info
    for step in range(max_steps):
        with torch.no_grad():
            action = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        action = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        deviations.append(abs(env.agent.navigation.current_lateral))
        last_info = info
        if terminated or truncated:
            break

    result = EpisodeResult(
        scenario_index=label if label is not None else seed,
        success=bool(last_info.get(TerminationState.SUCCESS, False)),
        collision=bool(last_info.get(TerminationState.CRASH, False)),
        off_road=bool(last_info.get(TerminationState.OUT_OF_ROAD, False)),
        max_step=bool(last_info.get(TerminationState.MAX_STEP, False)),
        route_completion=float(last_info.get("route_completion", 0.0)),
        mean_route_deviation=float(np.mean(deviations)) if deviations else 0.0,
        num_steps=step + 1,
    )
    if return_trace:
        return result, deviations
    return result


def evaluate(ckpt_path=CKPT_PATH, config_path=None, device="cpu", use_extra_dataset=True):
    """Closed-loop eval over the cleaned nuScenes eval range (degenerate scenarios excluded) plus,
    by default, the bundled Waymo scenarios as a second disjoint held-out set (Phase 4.5 step 4).
    """
    model, model_full_cfg = load_policy(ckpt_path, device)
    cfg = load_config(config_path) if config_path else load_config()
    split = cfg["scenario_split"]
    exclude = set(split.get("eval_exclude_indices", []))
    eval_indices = [i for i in range(split["eval_start"], split["eval_start"] + split["eval_num"])
                    if i not in exclude]

    horizon = cfg.get("horizon", 1000)
    episodes = []

    env = make_eval_env(config_path) if config_path else make_eval_env()
    try:
        for idx in eval_indices:
            result = run_episode(env, model, seed=idx, max_steps=horizon, device=device,
                                  label=f"nuscenes:{idx}")
            episodes.append(result)
            print(f"nuscenes:{idx}: success={result.success} collision={result.collision} "
                  f"off_road={result.off_road} route_completion={result.route_completion*100:.1f}% "
                  f"steps={result.num_steps}")
    finally:
        env.close()

    if use_extra_dataset:
        env_extra = make_eval_env_extra(config_path) if config_path else make_eval_env_extra()
        extra_num = cfg.get("eval_extra_num", 3)
        try:
            for idx in range(extra_num):
                result = run_episode(env_extra, model, seed=idx, max_steps=horizon, device=device,
                                      label=f"waymo:{idx}")
                episodes.append(result)
                print(f"waymo:{idx}: success={result.success} collision={result.collision} "
                      f"off_road={result.off_road} route_completion={result.route_completion*100:.1f}% "
                      f"steps={result.num_steps}")
        finally:
            env_extra.close()

    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "metrics.json"))
    args = parser.parse_args()

    episodes = evaluate(ckpt_path=args.ckpt)
    metrics_by_dataset = aggregate_metrics_by_dataset(episodes)
    for dataset, metrics in metrics_by_dataset.items():
        print(f"\n--- {dataset} ---")
        print_metrics_table(metrics)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_by_dataset, f, indent=2)
    print(f"\nSaved per-dataset metrics to {out_path}")
