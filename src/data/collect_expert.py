"""Collect (observation, expert_action) pairs from the logged ego trajectory of each
training-split scenario, for behavior cloning.

History (see ROADMAP.md Phase 4.5 / docs/architecture.md): the first version of this module
inverted a kinematic bicycle model from the logged teleport-replay trajectory to get pseudo-actions
after the fact. Phase 4.5's decisive test showed those labels do not reliably reproduce the logged
route through real physics (49.9% mean route completion across training scenarios, half ending in
collision) -- a label/dynamics mismatch, not covariate shift. `_pseudo_actions_for_track` and
`_wheelbase_and_max_steering` are kept below only because `src/eval/diagnose.py` still references
them to reproduce that historical check.

The active collection method now uses `src/data/tracking_expert.py`'s closed-loop pure-pursuit +
PID controller: it drives the ego through `make_train_env()` (default `EnvInputPolicy`, i.e. real
physics) every step, reacting to the vehicle's *actual* current state rather than the logged one.
The recorded action IS the action that was applied -- physically consistent by construction, and
defined at any state the car reaches (not just points on the logged path), which is what makes
DAgger (Phase 6a) meaningful later.

obs[t] is paired with the action the controller took at step t.
"""
import argparse
from pathlib import Path

import numpy as np

from metadrive.utils.math import compute_angular_velocity, wrap_to_pi
from src.data.tracking_expert import build_controller_for_scenario
from src.env.make_env import make_expert_collect_env, make_train_env, load_config

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ACCEL_SCALE_MPS2 = 3.0  # heuristic normalization scale for throttle_brake, see module docstring (legacy)


def _pseudo_actions_for_track(track, dt=0.1):
    """Vectorized [steering, throttle_brake] for every transition state[t] -> state[t+1]."""
    state = track["state"]
    position = np.asarray(state["position"])[:, :2]
    heading = np.asarray(state["heading"])
    velocity = np.asarray(state["velocity"])
    valid = np.asarray(state["valid"]).astype(bool)

    speed = np.linalg.norm(velocity, axis=1)
    n = len(position)

    yaw_rate = np.zeros(n - 1, dtype=np.float64)
    for t in range(n - 1):
        yaw_rate[t] = compute_angular_velocity(heading[t], heading[t + 1], dt)

    speed_t = speed[:-1]
    speed_safe = np.where(speed_t > 0.5, speed_t, np.inf)  # below ~0.5 m/s, curvature is ill-defined
    curvature = yaw_rate / speed_safe

    return position, heading, speed, valid, curvature


def _wheelbase_and_max_steering(vehicle):
    wheelbase = vehicle.FRONT_WHEELBASE + vehicle.REAR_WHEELBASE
    max_steering_rad = np.deg2rad(vehicle.max_steering)
    return wheelbase, max_steering_rad


def collect_scenario_legacy_heuristic(env, seed, max_steps):
    """Original open-loop kinematic-inversion collector. Kept only for reference -- see module
    docstring; superseded by collect_scenario's closed-loop tracking-controller approach."""
    obs, info = env.reset(seed=seed)
    scenario = env.engine.data_manager.current_scenario
    sdc_id = str(scenario["metadata"]["sdc_id"])
    track = scenario["tracks"][sdc_id]

    position, heading, speed, valid, curvature = _pseudo_actions_for_track(track)
    wheelbase, max_steering_rad = _wheelbase_and_max_steering(env.agent)

    obs_list, act_list = [], []
    n_transitions = min(len(position) - 1, max_steps)
    for t in range(n_transitions):
        if not (valid[t] and valid[t + 1]):
            obs, r, terminated, truncated, info = env.step([0.0, 0.0])
            if terminated or truncated:
                break
            continue

        steering = np.clip(np.arctan(wheelbase * curvature[t]) / max_steering_rad, -1.0, 1.0)
        accel = (speed[t + 1] - speed[t]) / 0.1
        throttle_brake = np.clip(accel / ACCEL_SCALE_MPS2, -1.0, 1.0)

        obs_list.append(obs)
        act_list.append([float(steering), float(throttle_brake)])

        obs, r, terminated, truncated, info = env.step([0.0, 0.0])  # action ignored by ReplayEgoCarPolicy
        if terminated or truncated:
            break

    return obs_list, act_list


def collect_scenario(env, seed, max_steps):
    """Drive the ego with the pure-pursuit + PID tracking controller through real physics,
    recording (obs, action) at every step. The action recorded is the action applied."""
    obs, info = env.reset(seed=seed)
    controller = build_controller_for_scenario(env)
    controller.reset()

    obs_list, act_list = [], []
    last_info = info
    for t in range(max_steps):
        action = controller.act(env.agent.position, env.agent.heading_theta, env.agent.speed)
        obs_list.append(obs)
        act_list.append(action.tolist())

        obs, r, terminated, truncated, info = env.step(action)
        last_info = info
        if terminated or truncated:
            break

    return obs_list, act_list, last_info


def collect_dataset(config_path=None, max_steps_per_scenario=1000):
    cfg = load_config(config_path) if config_path else load_config()
    split = cfg["scenario_split"]
    train_indices = list(range(split["train_start"], split["train_start"] + split["train_num"]))

    env = make_train_env(config_path) if config_path else make_train_env()
    all_obs, all_act, scenario_lengths = [], [], {}
    try:
        for idx in train_indices:
            obs_list, act_list, last_info = collect_scenario(env, seed=idx, max_steps=max_steps_per_scenario)
            all_obs.extend(obs_list)
            all_act.extend(act_list)
            scenario_lengths[idx] = len(obs_list)
            print(f"scenario {idx}: {len(obs_list)} transitions, "
                  f"route_completion={last_info.get('route_completion', 0)*100:.1f}%, "
                  f"crash={last_info.get('crash', False)}, out_of_road={last_info.get('out_of_road', False)}")
    finally:
        env.close()

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    act_arr = np.asarray(all_act, dtype=np.float32)
    return obs_arr, act_arr, train_indices, scenario_lengths


def print_summary(obs_arr, act_arr, train_indices):
    print("\n=== BC dataset summary ===")
    print(f"#scenarios collected: {len(train_indices)} (indices {train_indices})")
    print(f"#transitions: {obs_arr.shape[0]}")
    print(f"obs shape: {obs_arr.shape}")
    print(f"action shape: {act_arr.shape}")
    steering, throttle_brake = act_arr[:, 0], act_arr[:, 1]
    print(f"steering: mean={steering.mean():.4f} std={steering.std():.4f} "
          f"min={steering.min():.4f} max={steering.max():.4f}")
    print(f"throttle_brake: mean={throttle_brake.mean():.4f} std={throttle_brake.std():.4f} "
          f"min={throttle_brake.min():.4f} max={throttle_brake.max():.4f} "
          f"frac_positive={float((throttle_brake > 0).mean()):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "bc_dataset.npz"))
    parser.add_argument("--max_steps_per_scenario", type=int, default=1000)
    args = parser.parse_args()

    obs_arr, act_arr, train_indices, scenario_lengths = collect_dataset(
        max_steps_per_scenario=args.max_steps_per_scenario
    )
    print_summary(obs_arr, act_arr, train_indices)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_arr, act=act_arr)
    print(f"\nSaved dataset to {out_path}")
