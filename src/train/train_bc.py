"""Behavior cloning trainer: fit MLPPolicy on (obs, pseudo-action) pairs via MSE.

VRAM hygiene per ROADMAP.md: small batch, AMP autocast + GradScaler, pin_memory. The state-vector
MLP is tiny so this is mostly a demonstration of good habits -- watch nvidia-smi during the first
epoch and halve the batch size if it approaches 4 GB (it should not, on this model).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.env.make_env import load_config
from src.models.mlp_policy import MLPPolicy

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "bc_dataset.npz"
OUTPUTS_DIR = ROOT / "outputs"


def load_dataset(path, val_fraction, seed=0):
    data = np.load(path)
    obs, act = data["obs"], data["act"]
    n = len(obs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (obs[train_idx], act[train_idx]), (obs[val_idx], act[val_idx])


def make_loader(obs, act, batch_size, shuffle, device):
    ds = TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(act).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=(device.type == "cuda"))


def run_epoch(model, loader, optimizer, scaler, device, loss_fn, train):
    model.train(train)
    total_loss, n_samples = 0.0, 0
    for obs, act in loader:
        obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                pred = model(obs)
                loss = loss_fn(pred, act)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * obs.shape[0]
        n_samples += obs.shape[0]
    return total_loss / n_samples


def train(config_path=None, data_path=DATA_PATH, out_dir=OUTPUTS_DIR,
          ckpt_name="bc_best.pt", loss_plot_name="bc_loss.png"):
    cfg = load_config(config_path) if config_path else load_config()
    train_cfg, model_cfg = cfg["train"], cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"

    (train_obs, train_act), (val_obs, val_act) = load_dataset(data_path, train_cfg["val_fraction"])
    print(f"train: {len(train_obs)} transitions, val: {len(val_obs)} transitions")

    train_loader = make_loader(train_obs, train_act, train_cfg["batch_size"], shuffle=True, device=device)
    val_loader = make_loader(val_obs, val_act, train_cfg["batch_size"], shuffle=False, device=device)

    model = MLPPolicy(obs_dim=train_obs.shape[1], act_dim=train_act.shape[1],
                       hidden_sizes=tuple(model_cfg["hidden_sizes"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg.get("weight_decay", 0.0))
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    loss_fn = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_best = 0
    patience = train_cfg.get("early_stop_patience")
    train_losses, val_losses = [], []

    for epoch in range(train_cfg["epochs"]):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, loss_fn, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, scaler, device, loss_fn, train=False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch+1}/{train_cfg['epochs']}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_best = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss},
                out_dir / ckpt_name,
            )
        else:
            epochs_since_best += 1

        if device.type == "cuda" and epoch == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"peak VRAM after epoch 1: {peak_mb:.1f} MiB")

        if patience is not None and epochs_since_best >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement in {patience} epochs)")
            break

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("BC training/validation loss")
    plt.savefig(out_dir / loss_plot_name)
    print(f"Saved loss curve to {out_dir / loss_plot_name}")
    print(f"Best val loss: {best_val:.5f}, checkpoint at {out_dir / ckpt_name}")

    if device.type == "cuda":
        print(f"Peak VRAM (training run): {torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--out_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--ckpt_name", type=str, default="bc_best.pt")
    parser.add_argument("--loss_plot_name", type=str, default="bc_loss.png")
    args = parser.parse_args()
    train(data_path=args.data, out_dir=args.out_dir, ckpt_name=args.ckpt_name, loss_plot_name=args.loss_plot_name)
