"""Builders for the train/eval ScenarioEnv used throughout this project.

Scenario indices are split into a contiguous training range and a disjoint,
contiguous evaluation range within the bundled dataset (see configs/default.yaml).
The eval range is never used for expert data collection (Phase 2) -- only for
closed-loop evaluation (Phase 4).
"""
import yaml
from pathlib import Path

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _data_directory(dataset):
    return AssetLoader.file_path(AssetLoader.asset_path, dataset, unix_style=False)


def _build_env(cfg, start_scenario_index, num_scenarios, extra_config=None):
    env_config = dict(
        data_directory=_data_directory(cfg["dataset"]),
        start_scenario_index=start_scenario_index,
        num_scenarios=num_scenarios,
        sequential_seed=True,
        use_render=False,
        horizon=cfg.get("horizon", 1000),
    )
    if extra_config:
        env_config.update(extra_config)
    return ScenarioEnv(env_config)


def make_train_env(config_path=CONFIG_PATH, extra_config=None):
    """Env over the training scenario range, controlled by the policy's own actions."""
    cfg = load_config(config_path)
    split = cfg["scenario_split"]
    return _build_env(cfg, split["train_start"], split["train_num"], extra_config)


def make_eval_env(config_path=CONFIG_PATH, extra_config=None):
    """Env over the held-out evaluation scenario range, controlled by the policy's own actions."""
    cfg = load_config(config_path)
    split = cfg["scenario_split"]
    return _build_env(cfg, split["eval_start"], split["eval_num"], extra_config)


def make_eval_env_extra(config_path=CONFIG_PATH, extra_config=None):
    """Held-out env over a second bundled dataset (default: waymo), added in Phase 4.5 so the
    eval set isn't just a handful of scenarios from one dataset. Entirely disjoint from training.
    """
    cfg = load_config(config_path)
    dataset = cfg.get("eval_extra_dataset", "waymo")
    num = cfg.get("eval_extra_num", 3)
    env_config = dict(
        data_directory=_data_directory(dataset),
        start_scenario_index=0,
        num_scenarios=num,
        sequential_seed=True,
        use_render=False,
        horizon=cfg.get("horizon", 1000),
    )
    if extra_config:
        env_config.update(extra_config)
    return ScenarioEnv(env_config)


def make_expert_collect_env(config_path=CONFIG_PATH, extra_config=None):
    """Training-range env with the ego forced onto the logged replay policy.

    Used only by Phase 2 expert data collection -- never on the eval range.
    """
    cfg = load_config(config_path)
    split = cfg["scenario_split"]
    replay_config = {"agent_policy": ReplayEgoCarPolicy}
    if extra_config:
        replay_config.update(extra_config)
    return _build_env(cfg, split["train_start"], split["train_num"], replay_config)


if __name__ == "__main__":
    env = make_train_env()
    obs, info = env.reset(seed=0)
    print("obs type:", type(obs), "shape:", getattr(obs, "shape", None))
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    env.close()
