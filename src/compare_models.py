"""
KalmanNet vs Particle Filter vs Extended Kalman Filter
=====================================================
Self-contained benchmark script. It evaluates any number of **KalmanNet**
checkpoints together with **Particle Filter** (PF) and **Extended Kalman
Filter** (EKF) baselines on the 48-hour blood-glucose *test* set. For six
pre-defined initial-state scenarios it

* computes per-sequence RMSE and wall-clock inference time,
* runs paired *t*-tests and Cohen’s *d* for both metrics,
* prints tidy tables to the console, and
* writes a timestamped JSON bundle with all raw metrics, summaries, and
  statistics.

### Automatic training when no checkpoints are present
If **no KalmanNet checkpoints are found**, the script tries to train a new
KalmanNet model (a tiny GRU that predicts the Kalman gain):

1. Looks for `train.npz` in the same folder as `test.npz`.
2. If `train.npz` is missing, falls back to using the *test set itself* as
   training data (you’ll get a loud warning; scores will be optimistic).

The trained model is saved as `knet_autotrained.pth` in the output directory
and included in the benchmark run.

---------------------------------------------------------------------------
Usage examples
---------------------------------------------------------------------------
Run with every `*.pth` found in the default folder:
    python compare_models.py

Run with explicit checkpoints and a larger PF:
    python compare_models.py \
        --ckpts ./EKFvsKNvsPF/knet_standard.pth ./EKFvsKNvsPF/knet_mixed.pth \
        --pf_N 8000 \
        --out  ./EKFvsKNvsPF/results
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel


# ---------------------------------------------------------------------------
# 1.  Global configuration & paths
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
SEED   = 42

torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
torch.manual_seed(SEED)

Q_STD     = 0.05                            # process-noise σ (mmol/L·√min)
B_LAPLACE = 0.20                            # Laplace sensor-noise *b* (σ≈b√2)
R_GAUSS   = (B_LAPLACE * np.sqrt(2.0)) ** 2

OUTPUT_DIR = Path.cwd() / "EKFvsKNvsPF"
OUTPUT_DIR.mkdir(exist_ok=True)

# If a previous test folder exists, prefer its data; else use default
_llt = Path.cwd() / "last_last_test" / "data_knet"
DATA_DIR = _llt if _llt.exists() else OUTPUT_DIR / "data_knet"


# ---------------------------------------------------------------------------
# 2.  Scenario initial vectors
# ---------------------------------------------------------------------------
INIT_VECTORS = np.array([
    [5.5, 5.5, 81.4, 119.438, 0.0, 0.0],   # ±10 % up
    [4.5, 4.5, 66.6,  97.722, 0.0, 0.0],   # ±10 % down
    [6.0, 6.0, 88.8, 130.296, 0.0, 0.0],   # ±20 % up
    [4.0, 4.0, 59.2,  86.864, 0.0, 0.0],   # ±20 % down
    [6.5, 6.5, 96.2, 141.154, 0.0, 0.0],   # ±30 % up
    [3.5, 3.5, 51.8,  76.006, 0.0, 0.0],   # ±30 % down
], dtype=np.float32)

INIT_LABELS = [
    "±10 % up", "±10 % down",
    "±20 % up", "±20 % down",
    "±30 % up", "±30 % down",
]


# ---------------------------------------------------------------------------
# 3.  KalmanNet definition & training utilities
# ---------------------------------------------------------------------------
DEFAULT_HP = {
    "hidden": 64,
    "layers": 1,
    "act":    "relu",
    "lr":     6.614493869082084e-4,
    "drop":   0.6403746766700992,
    "bs":     64,
    "seq":    100,        # not used – we feed full sequences
    "epochs": 15,
    "opt":    "rmsprop",
    "clip":   0.5,
}


class GainGRU(nn.Module):
    """Tiny GRU → scalar Kalman gain (0–1)."""

    def __init__(self,
                 hidden_size: int = 32,
                 num_layers:  int = 1,
                 dropout:     float = 0.0,
                 act: str = "sigmoid") -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_size, 1)
        self.act = {"linear": nn.Identity(),
                    "relu":   nn.ReLU(),
                    "sigmoid": nn.Sigmoid()}[act]

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B,T,1)  → returns (B,T)
        out, _ = self.gru(z_seq)
        return self.act(self.fc(out)).squeeze(-1)


Q_MAT = torch.tensor([[Q_STD ** 2]], dtype=DTYPE, device=DEVICE)


@torch.no_grad()
def _kf_loop(K: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Fast scalar-KF loop (vectorised over batch)."""
    B, T = z.shape
    x_hat = z[:, :1]                                     # initial state
    P     = torch.full((B, 1, 1), 5.0, dtype=z.dtype, device=z.device)
    est   = torch.empty_like(z)
    for t in range(T):
        zt, Kt  = z[:, t:t+1], K[:, t:t+1]
        P_pred  = P + Q_MAT
        x_hat   = x_hat + Kt * (zt - x_hat)
        P       = (1 - Kt.unsqueeze(2)) * P_pred
        est[:, t] = x_hat.squeeze(1)
    return est


@torch.no_grad()
def make_knet_predictor(ckpt: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load a checkpoint and return a numpy-friendly predictor."""
    data = torch.load(ckpt, map_location=DEVICE)
    hp   = data.get("hyperparams", DEFAULT_HP)
    net  = GainGRU(hp["hidden"], hp["layers"], hp["drop"], hp["act"]).to(DEVICE)
    net.load_state_dict(data["state_dict"], strict=False)
    net.eval()

    def _predict(z_np: np.ndarray) -> np.ndarray:
        z_t = torch.as_tensor(z_np, dtype=DTYPE, device=DEVICE)\
                 .unsqueeze(0).unsqueeze(-1)             # (1,T,1)
        K   = net(z_t)                                   # (1,T)
        return _kf_loop(K, z_t.squeeze(-1)).squeeze(0).cpu().numpy()
    return _predict


# ---------- small helper for optimiser -------------------------------------
def _get_optimizer(params, hp):
    name = hp["opt"].lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=hp["lr"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=hp["lr"], momentum=0.9)
    return torch.optim.RMSprop(params, lr=hp["lr"])


def train_knet(z_train: np.ndarray,
               x_train: np.ndarray,
               out_path: Path,
               hp: Dict = DEFAULT_HP) -> Path:
    """Train KalmanNet from scratch and save it to *out_path*."""
    print(f"[INFO] Training KalmanNet from scratch → {out_path}")
    net   = GainGRU(hp["hidden"], hp["layers"], hp["drop"], hp["act"]).to(DEVICE)
    optim = _get_optimizer(net.parameters(), hp)
    crit  = nn.MSELoss()
    B     = hp["bs"]
    N     = len(z_train)

    for epoch in range(1, hp["epochs"] + 1):
        idx = np.random.permutation(N)
        ep_loss = 0.0
        for i in range(0, N, B):
            b = idx[i:i+B]
            z_b = torch.as_tensor(z_train[b], dtype=DTYPE, device=DEVICE).unsqueeze(-1)  # (B,T,1)
            x_b = torch.as_tensor(x_train[b], dtype=DTYPE, device=DEVICE)               # (B,T)
            K_hat = net(z_b)
            est   = _kf_loop(K_hat, z_b.squeeze(-1))
            loss  = crit(est, x_b)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp["clip"])
            optim.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= N
        print(f"  epoch {epoch:2d}/{hp['epochs']}  loss={ep_loss:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "hyperparams": hp}, out_path)
    print("[INFO] KalmanNet training finished.")
    return out_path


# ---------------------------------------------------------------------------
# 4.  Particle Filter & scalar EKF
# ---------------------------------------------------------------------------
class ParticleFilter:
    def __init__(self,
                 N: int,
                 x0: float,
                 q_std: float,
                 r_var: float,
                 spread: float = 0.5):
        self.N, self.q_std, self.r_var = N, q_std, r_var
        self.rng = np.random.default_rng(SEED)
        self.particles = x0 + spread * self.rng.standard_normal(N)
        self.weights   = np.full(N, 1.0 / N)

    def _resample(self):
        pos = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.searchsorted(np.cumsum(self.weights), pos)
        self.particles[:] = self.particles[idx]
        self.weights.fill(1.0 / self.N)

    def step(self, z: float) -> float:
        # predict
        self.particles += self.q_std * self.rng.standard_normal(self.N)
        # update
        like = np.exp(-0.5 * (z - self.particles) ** 2 / self.r_var)
        self.weights *= like
        self.weights += 1e-300
        self.weights /= self.weights.sum()
        # resample if needed
        if 1.0 / np.sum(self.weights ** 2) < 0.5 * self.N:
            self._resample()
        # state estimate
        return float(np.sum(self.weights * self.particles))


def pf_sequence(z_np: np.ndarray, N: int) -> np.ndarray:
    pf = ParticleFilter(N, x0=float(z_np[0]), q_std=Q_STD, r_var=R_GAUSS)
    return np.array([pf.step(float(zk)) for zk in z_np], dtype=np.float32)


def ekf_sequence(z_np: np.ndarray, x0: float) -> np.ndarray:
    x, P = x0, 5.0
    out  = np.empty_like(z_np)
    for k, zk in enumerate(z_np):
        P += Q_STD ** 2
        K   = P / (P + R_GAUSS)
        x  += K * (zk - x)
        P   = (1 - K) * P
        out[k] = x
    return out


# ---------------------------------------------------------------------------
# 5.  Statistics helpers
# ---------------------------------------------------------------------------
rmse      = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
cohen_d   = lambda a, b: float(np.mean(a - b) / np.std(a - b, ddof=1))
paired    = lambda a, b: {
    "t": float(ttest_rel(a, b)[0]),
    "p": float(ttest_rel(a, b)[1]),
    "d": cohen_d(a, b),
}


# ---------------------------------------------------------------------------
# 6.  Scenario evaluation
# ---------------------------------------------------------------------------
def run_scenario(initial_bg: float,
                 predictors: Dict[str, Callable[[np.ndarray], np.ndarray]],
                 z: np.ndarray,
                 x: np.ndarray):
    metrics = {m: {"rmse": [], "time": []} for m in predictors}
    for z_seq, x_seq in zip(z, x):
        z_mod = z_seq.copy()
        z_mod[0] = max(0.0, initial_bg)
        for name, fn in predictors.items():
            t0  = time.perf_counter()
            est = fn(z_mod)
            dur = time.perf_counter() - t0
            metrics[name]["rmse"].append(rmse(est, x_seq))
            metrics[name]["time"].append(dur)
    return metrics


summarise = lambda m: {
    k: {
        "rmse_mean": float(np.mean(v["rmse"])),
        "rmse_std":  float(np.std(v["rmse"], ddof=1)),
        "time_mean": float(np.mean(v["time"])),
        "time_std":  float(np.std(v["time"], ddof=1)),
    }
    for k, v in m.items()
}


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR,
                        help="directory containing test.npz "
                             "(and optionally train.npz) {z,x}")
    parser.add_argument("--ckpts", type=Path, nargs="*",
                        default=[], help="KalmanNet checkpoint paths (*.pth)")
    parser.add_argument("--pf_N",  type=int, default=5000,
                        help="particle count for PF")
    parser.add_argument("--out",   type=Path,
                        default=OUTPUT_DIR / "results",
                        help="output JSON file or directory")
    args = parser.parse_args()

    # ---- resolve checkpoints ------------------------------------------------
    ckpts: List[Path] = list(args.ckpts)
    if not ckpts:
        ckpts = sorted(OUTPUT_DIR.glob("*.pth"))
        if ckpts:
            print(f"[INFO] Auto-loaded {len(ckpts)} checkpoint(s) from {OUTPUT_DIR}.")

    # ---- load test set ------------------------------------------------------
    data_file = args.data_dir / "test.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing test set: {data_file}")
    with np.load(data_file) as d:
        z_test, x_test = d["z"], d["x"]
    n_seq, T = z_test.shape
    print(f"Loaded test set – {n_seq} sequences × {T} steps.")

    # ---- optional training if no ckpts -------------------------------------
    if not ckpts:
        print("[WARN] No KalmanNet checkpoints found – starting on-the-fly training.")
        train_file = args.data_dir / "train.npz"
        if train_file.exists():
            with np.load(train_file) as d:
                z_train, x_train = d["z"], d["x"]
        else:
            print("[WARN] train.npz not found – "
                  "using test set itself for training (expect optimistic RMSE).")
            z_train, x_train = z_test, x_test
        ckpt_path = OUTPUT_DIR / "knet_autotrained.pth"
        train_knet(z_train, x_train, ckpt_path)
        ckpts = [ckpt_path]

    # ---- build predictors ---------------------------------------------------
    predictors: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Particle": lambda z, N=args.pf_N: pf_sequence(z, N),
        "EKF":      lambda z: ekf_sequence(z, x0=float(z[0])),
    }
    for ck in ckpts:
        predictors[ck.stem] = make_knet_predictor(ck)

    # ---- scenario loop ------------------------------------------------------
    results: Dict[str, Dict] = {}
    for lbl, vec in zip(INVOKE_LABELS := INIT_LABELS, INIT_VECTORS):
        init_bg = float(vec[0])
        print(f"== Scenario {lbl} (BG₀={init_bg:.2f} mmol/L) ==")
        metrics = run_scenario(init_bg, predictors, z_test, x_test)
        summary = summarise(metrics)
        stats   = {}
        for a, b in combinations(predictors.keys(), 2):
            stats[f"{a}_vs_{b}"] = {
                "rmse": paired(np.array(metrics[a]["rmse"]),
                               np.array(metrics[b]["rmse"])),
                "time": paired(np.array(metrics[a]["time"]),
                               np.array(metrics[b]["time"])),
            }
        results[lbl] = {"summary": summary, "stats": stats}

        # console table
        print("Model           RMSE ± SD        Time ± SD (s)")
        for m, s in summary.items():
            print(f"{m:<15}"
                  f"{s['rmse_mean']:.4f} ± {s['rmse_std']:.4f}   "
                  f"{s['time_mean']:.3f} ± {s['time_std']:.3f}")

    # ---- save JSON ----------------------------------------------------------
    out_path = args.out
    if out_path.suffix != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
