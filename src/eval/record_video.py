"""Render env rollouts to GIF (top-down, CPU renderer).

Two modes:
  - replay (default here): ego follows the logged trajectory (ReplayEgoCarPolicy).
    Used for Phase 1's sanity-check demo and Phase 5 failure clips of the dataset itself.
  - policy: a trained model drives the ego each step (used by Phase 4 closed-loop eval).
"""
import argparse
from pathlib import Path

import imageio
import numpy as np

from src.env.make_env import make_train_env, make_eval_env, make_eval_env_extra, load_config
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def record_replay_gif(out_path, seed=0, max_steps=300, fps=10, use_eval_split=False):
    """Roll out the logged ego trajectory (the BC expert) and save a top-down GIF."""
    env_fn = make_eval_env if use_eval_split else make_train_env
    env = env_fn(extra_config={"agent_policy": ReplayEgoCarPolicy})
    frames = []
    try:
        env.reset(seed=seed)
        for _ in range(max_steps):
            _, _, terminated, truncated, _ = env.step([0.0, 0.0])  # action ignored under ReplayEgoCarPolicy
            frame = env.render(mode="top_down", film_size=(800, 800))
            frames.append(np.asarray(frame))
            if terminated or truncated:
                break
    finally:
        env.close()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path, len(frames)


TIGHT_RENDER = dict(
    film_size=(2000, 2000),              # large canvas covers any map
    scaling=8,                           # 8 px/m → screen shows 75 m around ego at 600 px
    screen_size=(600, 600),              # output frame size
    target_agent_heading_up=True,        # ego always points up for clean framing
    draw_target_vehicle_trajectory=True, # solid teal/green trail marks the ego's path,
                                         # making it unmistakable in the top-down view
)
WIDE_RENDER = dict(film_size=(800, 800))  # original wide-map view kept for compatibility


def record_policy_gif(out_path, model, seed, max_steps=300, fps=10, device="cpu",
                       dataset="nuscenes", render_kwargs=None):
    """Roll out a trained model driving the ego and save a top-down GIF.

    render_kwargs are passed to env.render() on every step; the first call constructs the
    TopDownRenderer with those params (film_size, scaling, screen_size, target_agent_heading_up).
    Defaults to TIGHT_RENDER — ego-tracked, zoomed in, heading-up — which produces the most
    visually compelling showcase output.
    """
    import torch
    if render_kwargs is None:
        render_kwargs = TIGHT_RENDER
    env = make_eval_env_extra() if dataset == "waymo" else make_eval_env()
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            with torch.no_grad():
                action = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render(mode="top_down", **render_kwargs)
            frames.append(np.asarray(frame))
            if terminated or truncated:
                break
    finally:
        env.close()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path, len(frames)


def record_policy_rollouts(ckpt_path, seeds, out_dir=OUTPUTS_DIR, max_steps=300, fps=10, device="cpu",
                            dataset="nuscenes", prefix="rollout"):
    """Render one <prefix>_<dataset>_<seed>.gif per held-out scenario seed under the trained policy."""
    from src.eval.closed_loop_eval import load_policy
    model, _ = load_policy(ckpt_path, device=device)
    paths = []
    for seed in seeds:
        out_path = Path(out_dir) / f"{prefix}_{dataset}_{seed}.gif"
        path, n_frames = record_policy_gif(out_path, model, seed=seed, max_steps=max_steps, fps=fps,
                                            device=device, dataset=dataset)
        print(f"{dataset}:{seed}: saved {n_frames} frames to {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "policy"], default="replay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds for --mode policy")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "demo_replay.gif"))
    parser.add_argument("--ckpt", type=str, default=str(Path(OUTPUTS_DIR) / "bc_best.pt"))
    parser.add_argument("--dataset", choices=["nuscenes", "waymo"], default="nuscenes")
    parser.add_argument("--prefix", type=str, default="rollout")
    args = parser.parse_args()

    if args.mode == "replay":
        path, n_frames = record_replay_gif(args.out, seed=args.seed, max_steps=args.max_steps)
        print(f"Saved {n_frames} frames to {path}")
    else:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
        record_policy_rollouts(args.ckpt, seeds, max_steps=args.max_steps, dataset=args.dataset, prefix=args.prefix)
