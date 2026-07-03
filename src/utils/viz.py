"""Phase 5 headline figure: the debugging arc.

Open-loop validation loss improves monotonically across all three pipeline stages, but closed-loop
route completion does NOT -- it gets *worse* after fixing the labels (stage i -> ii) before DAgger
partially recovers it (stage ii -> iii). That single picture is the project's headline finding: an
improving one-step loss does not imply an improving driving policy.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"

STAGES = ["(i) heuristic\nlabels", "(ii) reactive-expert\nlabels (BC only)", "(iii) BC +\n1 DAgger iter"]
# Open-loop val loss read from each checkpoint / docs/architecture.md (see Phase 5 notes for the
# heuristic-label number, whose checkpoint was since overwritten by later retrains).
VAL_LOSSES = [0.1405, 0.0678, 0.0642]


def make_headline_figure(metrics_paths=("outputs/metrics_before_fix.json", "outputs/metrics.json",
                                         "outputs/metrics_dagger.json"),
                          out_path=None):
    out_path = Path(out_path) if out_path else OUTPUTS_DIR / "headline_figure.png"

    overall_rc, nuscenes_rc, waymo_rc, off_road = [], [], [], []
    for p in metrics_paths:
        d = json.load(open(ROOT / p))
        if "overall" in d:
            overall_rc.append(d["overall"]["mean_route_completion"] * 100)
            off_road.append(d["overall"]["off_road_rate"] * 100)
            nuscenes_rc.append(d["nuscenes"]["mean_route_completion"] * 100)
            waymo_rc.append(d["waymo"]["mean_route_completion"] * 100)
        else:
            # stage (i) was saved before per-dataset aggregation existed; recompute from per_episode.
            overall_rc.append(d["mean_route_completion"] * 100)
            off_road.append(d["off_road_rate"] * 100)
            nu = [e for e in d["per_episode"] if str(e["scenario_index"]).startswith("nuscenes")]
            wa = [e for e in d["per_episode"] if str(e["scenario_index"]).startswith("waymo")]
            nuscenes_rc.append(100 * np.mean([e["route_completion"] for e in nu]))
            waymo_rc.append(100 * np.mean([e["route_completion"] for e in wa]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(STAGES, VAL_LOSSES, marker="o", color="tab:blue")
    ax.set_ylabel("open-loop val MSE loss")
    ax.set_title("Open-loop loss: monotonically improves")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(STAGES))
    width = 0.25
    ax.bar(x - width, nuscenes_rc, width, label="nuScenes")
    ax.bar(x, waymo_rc, width, label="Waymo")
    ax.bar(x + width, overall_rc, width, label="overall", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("closed-loop mean route completion (%)")
    ax.set_title("Closed-loop: gets WORSE before DAgger recovers it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Open-loop loss does not predict closed-loop driving quality", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved headline figure to {out_path}")
    return out_path


if __name__ == "__main__":
    make_headline_figure()
