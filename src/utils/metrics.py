"""Closed-loop driving metrics computed from a single episode's `info` dict / vehicle state.

Per-episode signals come straight from the env (see metadrive/constants.py TerminationState and
ScenarioEnv's step_info): `route_completion` and the various crash/out-of-road flags. Route
deviation is `vehicle.navigation.current_lateral` -- the lateral offset from the logged reference
trajectory that TrajectoryNavigation already tracks every step (ScenarioEnv builds the navigation
route directly from the logged ego track, so this *is* "distance from the logged route").
"""
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class EpisodeResult:
    scenario_index: Union[int, str]  # e.g. "nuscenes:6" or "waymo:0" once multiple datasets are mixed
    success: bool
    collision: bool
    off_road: bool
    max_step: bool
    route_completion: float
    mean_route_deviation: float
    num_steps: int


def aggregate_metrics(episodes: List[EpisodeResult]) -> dict:
    n = len(episodes)
    return {
        "num_episodes": n,
        "success_rate": sum(e.success for e in episodes) / n,
        "collision_rate": sum(e.collision for e in episodes) / n,
        "off_road_rate": sum(e.off_road for e in episodes) / n,
        "max_step_rate": sum(e.max_step for e in episodes) / n,
        "mean_route_completion": sum(e.route_completion for e in episodes) / n,
        "mean_route_deviation": sum(e.mean_route_deviation for e in episodes) / n,
        "per_episode": [
            {
                "scenario_index": e.scenario_index,
                "success": e.success,
                "collision": e.collision,
                "off_road": e.off_road,
                "max_step": e.max_step,
                "route_completion": e.route_completion,
                "mean_route_deviation": e.mean_route_deviation,
                "num_steps": e.num_steps,
            }
            for e in episodes
        ],
    }


def aggregate_metrics_by_dataset(episodes: List[EpisodeResult]) -> dict:
    """Group episodes by the dataset prefix of their label (e.g. "nuscenes:6" -> "nuscenes") and
    aggregate separately, plus an "overall" pool. Falls back to a single "overall" group if labels
    aren't "dataset:index" strings.
    """
    groups = {}
    for e in episodes:
        if isinstance(e.scenario_index, str) and ":" in e.scenario_index:
            dataset = e.scenario_index.split(":", 1)[0]
        else:
            dataset = "overall"
        groups.setdefault(dataset, []).append(e)

    result = {"overall": aggregate_metrics(episodes)}
    for dataset, group_episodes in groups.items():
        result[dataset] = aggregate_metrics(group_episodes)
    return result


def print_metrics_table(metrics: dict):
    print("\n=== Closed-loop evaluation metrics ===")
    print(f"episodes:              {metrics['num_episodes']}")
    print(f"success rate:          {metrics['success_rate']*100:.1f}%")
    print(f"collision rate:        {metrics['collision_rate']*100:.1f}%")
    print(f"off-road rate:         {metrics['off_road_rate']*100:.1f}%")
    print(f"max-step (timeout):    {metrics['max_step_rate']*100:.1f}%")
    print(f"mean route completion: {metrics['mean_route_completion']*100:.1f}%")
    print(f"mean route deviation:  {metrics['mean_route_deviation']:.3f} m")
    print("\nper-episode:")
    for ep in metrics["per_episode"]:
        print(f"  scenario {ep['scenario_index']}: success={ep['success']} collision={ep['collision']} "
              f"off_road={ep['off_road']} route_completion={ep['route_completion']*100:.1f}% "
              f"route_deviation={ep['mean_route_deviation']:.3f}m steps={ep['num_steps']}")
