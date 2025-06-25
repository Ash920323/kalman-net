
# ---------------------------------------------------------------------------
#  Imports & global configuration
# ---------------------------------------------------------------------------
import os, sys, math, json, time
os.environ["TQDM_DISABLE"] = "1"
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from scipy.integrate import ode
from numpy.random import default_rng
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import optuna

# Project‑local filters -------------------------------------------------------
from util.filter import KalmanFilter, ParticleFilter

# -------- paths -------------------------------------------------------------
OUTPUT_DIR = Path.cwd() / "last_test"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR   = OUTPUT_DIR / "data_knet"
DATA_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE      = torch.float32
RNG        = default_rng(42)
VAL_FRAC   = 0.10               # validation split
MAX_EPOCHS = 30                 # upper bound during Optuna search
PRINT_EVERY = 1

# ---------------------------------------------------------------------------
# 1. Physio‑model parameters & helpers
# ---------------------------------------------------------------------------
Q_STD     = 0.05      # process noise σ
SIGMA_G   = 0.20      # Gaussian sensor σ
B_LAPLACE = 0.20      # Laplace sensor b parameter (σ≈b*√2)

params = {
    # Glucose subsystem (eq. 27)
    "PG": 0.022, "Si": 1.2e-4, "alpha_G": 1.0, "EGPb": 1.27,
    "CNS": 1.16, "VG": 10.0,
    # Insulin transport (eqs. 28–29)
    "nI": 0.157, "nC": 0.0159, "nK": 0.0165, "nL": 0.003,
    "alpha_I": 0.18, "VI": 11.0, "xL": 0.60,
    # Endogenous insulin (eq. 32)
    "k1": 210, "k2": 300, "k3": 0.007,
    # Stomach / gut (eqs. 30–31)
    "d1": 0.05, "d2": 0.06, "Pmax": 1.50, "PN": 0.011,
    # Interstitial glucose (eq. 34)
    "beta1": 0.002, "beta2": 0.02,
    # Basal insulin infusion
    "uex": 58.91,
}

MEAL_WINDOWS  = [(7*60, 11*60), (12*60, 14*60), (18*60, 21*60)]
MEAL_DURATION = 40
D_RATE        = 5 * (1000/180) / MEAL_DURATION  # mmol/min

def sample_meal_times(rng: np.random.Generator):
    while True:
        t1 = rng.integers(*MEAL_WINDOWS[0])
        t2 = rng.integers(*MEAL_WINDOWS[1])
        t3 = rng.integers(*MEAL_WINDOWS[2])
        if (t2 - t1) >= 5*60 and (t3 - t2) >= 5*60:
            return int(t1), int(t2), int(t3)

def make_D_of_t(meal_times):
    meal_times = np.asarray(meal_times)
    def D_of_t(t_min):
        t = np.asarray(t_min)
        active = ((t[:, None] if t.ndim else t) >= meal_times) & (
                  (t[:, None] if t.ndim else t) < (meal_times + MEAL_DURATION))
        return D_RATE * active.sum(axis=-1)
    return D_of_t

def glucose_insulin_model(t, x, p):
    BG, IG, Qs, I, P1, P2 = x
    P   = p["PN"] + min(p["d2"] * P2, p["Pmax"])
    uen = p["k1"] * np.exp(-I / p["k2"]) + p["k3"]

    dBG = (-p["PG"]*BG - p["Si"]*BG*Qs/(1+p["alpha_G"]*Qs) + P + p["EGPb"] - p["CNS"]) / p["VG"]
    dQs = p["nI"]*(I - Qs) - p["nC"]*Qs/(1+p["alpha_G"]*Qs)
    dI  = (-p["nK"]*I - p["nL"]*I/(1+p["alpha_I"]*I) - p["nI"]*(I-Qs)
           + p["uex"]/p["VI"] + (1-p["xL"]) * uen / p["VI"])
    dP1 = -p["d1"]*P1 + D_of_t(t)
    dP2 = -min(p["d2"]*P2, p["Pmax"]) + p["d1"]*P1
    dIG = p["beta1"]*BG - p["beta2"]*IG
    return [dBG, dIG, dQs, dI, dP1, dP2]

# ---------------------------------------------------------------------------
# 2. Dataset simulation (loads from cache if present)
# ---------------------------------------------------------------------------
t_end = 48*60; TS = np.arange(0, t_end+1, 1, dtype=np.int32); T = TS.size
x0_nom = np.array([5.0, 5.0, 74.0, 108.58, 0.0, 0.0], np.float32)

def simulate_bg_trace(meal_times):
    global D_of_t
    D_of_t = make_D_of_t(meal_times)
    solver = ode(lambda t,x: glucose_insulin_model(t,x,params)).set_integrator("dopri5")
    solver.set_initial_value(x0_nom, 0.0)
    xs = np.zeros((T,6), np.float32); xs[0]=x0_nom
    for k in range(1,T): xs[k]=solver.integrate(TS[k])
    return xs[:,0]

def build_split(N, mixed=False):
    bg_true = np.zeros((N, T), np.float32)
    z_lap   = np.zeros_like(bg_true)

    if sys.stdout.isatty():
        iterator = tqdm(range(N), desc=f"sim {N} {'mixed' if mixed else 'std'}", total=N,
                         ncols=80, miniters=max(1, N//10))
    else:
        iterator = range(N)

    for i in iterator:
        nm = RNG.integers(3, 6) if mixed else 3
        mt = np.sort(RNG.choice(t_end - 60, size=nm, replace=False))
        bg_true[i] = simulate_bg_trace(mt)
        z_lap[i]   = bg_true[i] + RNG.laplace(0.0, B_LAPLACE, bg_true[i].shape)

    if not sys.stdout.isatty():
        print(f"sim {N} {'mixed' if mixed else 'std'} done ✅")

    return z_lap, bg_true


files = {k: DATA_DIR/f"{k}.npz" for k in ("train_std","train_mix","test")}

if all(p.exists() for p in files.values()):
    print("✅ Found existing dataset files; loading …")
    def _load(path: Path):
        with np.load(path) as d:
            return d["z"], d["x"]        # ignore any extra arrays (e.g. "meal")
    z_train_std, x_train_std = _load(files["train_std"])
    z_train_mix, x_train_mix = _load(files["train_mix"])
    z_test,      x_test      = _load(files["test"     ])
else:
    print("⚙️  Dataset files missing; generating …")
    z_train_std, x_train_std = build_split(200, mixed=False)
    z_train_mix, x_train_mix = build_split(200, mixed=True)
    z_test,      x_test      = build_split(200, mixed=False)
    np.savez_compressed(files["train_std"], z=z_train_std, x=x_train_std)
    np.savez_compressed(files["train_mix"], z=z_train_mix, x=x_train_mix)
    np.savez_compressed(files["test"     ], z=z_test,      x=x_test)

class GlucoseDataset(torch.utils.data.Dataset):
    def __init__(self,z,x): self.z=torch.as_tensor(z,dtype=DTYPE); self.x=torch.as_tensor(x,dtype=DTYPE)
    def __len__(self): return len(self.z)
    def __getitem__(self,idx): return self.z[idx], self.x[idx]

train_std_ds = GlucoseDataset(z_train_std,x_train_std)
train_mix_ds = GlucoseDataset(z_train_mix,x_train_mix)
test_ds      = GlucoseDataset(z_test,x_test)

# ---------------------------------------------------------------------------
# 3.  KalmanNet (GRU) model & utilities
# ---------------------------------------------------------------------------
class GainGRU(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.0, output_activation: str = "sigmoid"):
        super().__init__()
        # -- GRU with correct keyword ordering ----------------------------
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                # <‑‑ always batch‑first
            dropout=dropout if num_layers > 1 else 0.0,
            bias=True,
        )
        # -- Output activation -------------------------------------------
        self.fc = nn.Linear(hidden_size, 1)
        self.act = {
            "linear":  nn.Identity(),
            "relu":    nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
        }.get(output_activation, nn.Sigmoid())

    def forward(self, z_seq):                 # z_seq: (B, T, 1)
        out, _ = self.gru(z_seq)
        return self.act(self.fc(out)).squeeze(-1)  # (B,T)(self,z): out,_=self.gru(z); return self.act(self.fc(out)).squeeze(-1)

Q_mat = torch.tensor([[Q_STD**2]], dtype=DTYPE, device=DEVICE)
@torch.jit.script
def kf_batch(K: torch.Tensor, z: torch.Tensor, Q: torch.Tensor):
    """Vectorised 1‑D Kalman filter (state‑noise Q, direct‑observation H=1)."""
    B, T = z.shape
    x_hat = z[:, 0].view(B, 1)
    P     = torch.full((B, 1, 1), 5.0, device=z.device)
    est   = torch.zeros(B, T, device=z.device)

    for t in range(T):
        zt = z[:, t].view(B, 1)
        Kt = K[:, t].view(B, 1)
        # predict
        x_pred = x_hat
        P_pred = P + Q
        # update
        x_hat = x_pred + Kt * (zt - x_pred)
        P     = (1 - Kt.unsqueeze(2)) * P_pred
        est[:, t] = x_hat.squeeze(1)
    return est

# Alias for backward‑compatibility with older helper functions
kalman_filter_batch = kf_batch

MSE = nn.MSELoss()

def crop_batch(z,x,len_):
    if z.size(1)==len_: return z,x
    s=torch.randint(0,z.size(1)-len_+1,(1,)).item(); return z[:,s:s+len_], x[:,s:s+len_]

def batch_loss(net,z_b,x_b,grad_clip=None,opt=None):
    K=net(z_b.unsqueeze(-1)); est=kf_batch(K,z_b,Q_mat); loss=MSE(est,x_b)
    if opt is not None:
        opt.zero_grad(); loss.backward();
        if grad_clip: nn.utils.clip_grad_norm_(net.parameters(),grad_clip)
        opt.step()
    return loss.item()

# ---------------------------------------------------------------------------
# 4.  Optuna hyper‑parameter search
# ---------------------------------------------------------------------------
HIDDEN_OPTS=[8,16,32,64]; NLAYER_OPTS=[1,2]; ACT_OPTS=["relu"]
LR_RANGE=(1e-5,1e-3); BATCH_OPTS=[32,64,128]
SEQ_OPTS=[50,100,200]; OPTIM_OPTS=["adam","rmsprop"]; CLIP_OPTS=[0.5,1.0]

def make_loader(ds,bs,shuf): return DataLoader(ds,bs,shuf,drop_last=False,num_workers=4)

def build_loaders(ds,bs):
    val_sz=int(len(ds)*VAL_FRAC); train_sz=len(ds)-val_sz
    tr,val=random_split(ds,[train_sz,val_sz],generator=torch.Generator().manual_seed(42))
    return make_loader(tr,bs,True), make_loader(val,bs,False)

def eval_val(net,dl):
    net.eval(); tot=0.0
    with torch.no_grad():
        for z,x in dl:
            z,x=z.to(DEVICE),x.to(DEVICE); K=net(z.unsqueeze(-1)); est=kf_batch(K,z,Q_mat); tot+=MSE(est,x).item()
    return tot/len(dl)

def objective(trial):
    hp={
        "hidden": trial.suggest_categorical("hidden",HIDDEN_OPTS),
        "layers": trial.suggest_categorical("layers",NLAYER_OPTS),
        "act":    trial.suggest_categorical("act",ACT_OPTS),
        "lr":     trial.suggest_float("lr",*LR_RANGE,log=True),
        "drop":   trial.suggest_float("drop", 0.0, 0.8),
        "bs":     trial.suggest_categorical("bs",BATCH_OPTS),
        "seq":    trial.suggest_categorical("seq",SEQ_OPTS),
        "opt":    trial.suggest_categorical("opt",OPTIM_OPTS),
        "clip":   trial.suggest_categorical("clip",CLIP_OPTS),
    }
    tr_dl,val_dl=build_loaders(train_std_ds,hp["bs"])
    net=GainGRU(hp["hidden"],hp["layers"],hp["drop"],hp["act"]).to(DEVICE)
    opt_cls=optim.Adam if hp["opt"]=="adam" else optim.RMSprop; opt=opt_cls(net.parameters(),lr=hp["lr"])
    best=float("inf")
    for epoch in range(1,MAX_EPOCHS+1):
        net.train()
        for z,x in tr_dl:
            z,x=z.to(DEVICE),x.to(DEVICE); z,x=crop_batch(z,x,hp["seq"]); batch_loss(net,z,x,hp["clip"],opt)
        val=eval_val(net,val_dl); trial.report(val,epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        best=min(best,val)
    return best

# ---------------------------------------------------------------------------
# 5.  Training helpers
# ---------------------------------------------------------------------------
@dataclass
class Args: trials:int=200; timeout:int=18000; persist:bool=False

def run_optimization(args:Args):
    storage="sqlite:///optuna_study.db" if args.persist else None
    study=optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner(2),storage=storage,load_if_exists=True,study_name="kalman_net")
    study.optimize(objective,n_trials=args.trials,timeout=args.timeout,show_progress_bar=True,gc_after_trial=True)
    (OUTPUT_DIR/"best_hparams.json").write_text(json.dumps(study.best_params,indent=2))
    return study.best_params

def train_final(ds, hp):
    tr_dl, val_dl = build_loaders(ds, hp["bs"])
    net = GainGRU(hp["hidden"],hp["layers"],hp["drop"],hp["act"]).to(DEVICE)
    opt_cls = optim.Adam if hp["opt"]=="adam" else optim.RMSprop
    opt = opt_cls(net.parameters(), lr=hp["lr"])

    train_losses, val_losses = [], []
    for epoch in range(1, MAX_EPOCHS+1):
        # --- training epoch ---
        net.train()
        epoch_loss = 0.0
        for z, x in tr_dl:
            z, x = z.to(DEVICE), x.to(DEVICE)
            epoch_loss += batch_loss(net, z, x, hp["clip"], opt)
        train_losses.append(epoch_loss / len(tr_dl))

        # --- validation loss ---
        val_loss = eval_val(net, val_dl)
        val_losses.append(val_loss)

    return net, train_losses, val_losses

@torch.no_grad()
def gru_filter(net,z_np):
    z_t=torch.tensor(z_np,dtype=DTYPE,device=DEVICE).view(1,-1,1)
    K=net(z_t).squeeze(-1); est=kf_batch(K,z_t.squeeze(-1),Q_mat)
    return est.cpu().numpy().reshape(-1)

def evaluate_test(net):
    net.eval()
    start=time.perf_counter(); preds=[gru_filter(net,z) for z in z_test]
    dur=time.perf_counter()-start; rmse=math.sqrt(np.mean((np.stack(preds)-x_test)**2))
    return rmse,dur

# ---------------------------------------------------------------------------
# 6.  Scenario evaluation helpers
# ---------------------------------------------------------------------------
INIT_VECTORS=np.array([
    [5.5,5.5,81.4,119.438,0.,0.], [4.5,4.5,66.6,97.722,0.,0.],
    [6.0,6.0,88.8,130.296,0.,0.], [4.0,4.0,59.2,86.864,0.,0.],
    [6.5,6.5,96.2,141.154,0.,0.], [3.5,3.5,51.8,76.006,0.,0.]],np.float32)
INIT_LABELS=["±10 % up","±10 % down","±20 % up","±20 % down","±30 % up","±30 % down"]

@torch.no_grad()
def apply_kalmannet(net,z_np,x0_scalar):
    z_mod=z_np.copy(); z_mod[0]=x0_scalar
    z_t=torch.tensor(z_mod,dtype=DTYPE,device=DEVICE).unsqueeze(0).unsqueeze(-1)
    K=net(z_t).squeeze(0); est=kf_batch(K.unsqueeze(0),torch.tensor(z_mod,dtype=DTYPE,device=DEVICE).unsqueeze(0),Q_mat)
    return est.squeeze(0).cpu().numpy()

def run_pf_sequence(z_np,x0_scalar,N=5000,init_spread=0.5):
    pf=ParticleFilter(N,1,lambda x,w:x+w,lambda x,v:x+v,
        lambda: np.random.normal(0.0,Q_STD,(1,)),
        lambda: np.random.laplace(0.0,B_LAPLACE,(1,)),
        lambda z,x: (1/(2*B_LAPLACE))*np.exp(-abs(z-x[0])/B_LAPLACE),
        init_particles=np.random.normal(x0_scalar,init_spread,size=(N,1)))
    est=np.empty_like(z_np)
    for k,z in enumerate(z_np): x_hat,_=pf.step(z); est[k]=x_hat.item()
    return est

def plot_loss_curves(train_losses, val_losses):
    """
    Plot training and validation loss vs. epochs, with y-axis 0–1.
    """
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses,  label='Training Loss')
    plt.plot(epochs, val_losses,    label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.ylim(0, 1)
    plt.xlim(1, len(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png")
# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    # If you haven't already, disable all tqdm bars at script startup:
    # import os
    # os.environ["TQDM_DISABLE"] = "1"

    print("Device:", DEVICE)
    best_hp = run_optimization(Args())  # make sure run_optimization was called with show_progress_bar=False

    # --- FIX: unpack train_final() so models[name] is the network itself ---
    models = {}
    for name, ds in {"standard": train_std_ds, "mixed": train_mix_ds}.items():
        net, train_losses, val_losses = train_final(ds, best_hp)
        models[name] = net
        # (if you need train_losses/val_losses later, stash them here)

    # Evaluate & save each model
    perf_records = []
    for name, net in models.items():
        torch.save(
            {"state_dict": net.state_dict(), "hparams": best_hp},
            OUTPUT_DIR / f"knet_{name}.pth"
        )
        rmse, dur = evaluate_test(net)
        perf_records.append((name, rmse, dur))

    df_perf = (
        pd.DataFrame(perf_records, columns=["Model", "RMSE", "Inference[s]"])
          .sort_values("RMSE")
    )
    df_perf.to_csv(OUTPUT_DIR / "test_performance.csv", index=False)
    print("\n=== Test performance ===\n", df_perf.to_string(index=False))

    # Plot loss curves & re‐evaluate if desired
    results = {}
    for name, ds in {"standard": train_std_ds, "mixed": train_mix_ds}.items():
        net, train_losses, val_losses = train_final(ds, best_hp)
        print(f"\n=== Loss curves for '{name}' model ===")
        plot_loss_curves(train_losses, val_losses)

        torch.save(
            {"state_dict": net.state_dict(), "hparams": best_hp},
            OUTPUT_DIR / f"knet_{name}.pth"
        )
        rmse, dur = evaluate_test(net)
        results[name] = (net, rmse, dur)

    # Choose best model for scenario analysis
    best_name = df_perf.iloc[0, 0]
    best_net  = models[best_name]

    scenario_rows = []
    for label, vec in zip(INIT_LABELS, INIT_VECTORS):
        x0_bg = float(vec[0])

        # KalmanNet
        start = time.perf_counter()
        kn_preds = [apply_kalmannet(best_net, z, x0_bg) for z in z_test]
        kn_time  = time.perf_counter() - start
        kn_rmse  = math.sqrt(np.mean((np.stack(kn_preds) - x_test) ** 2))

        # Particle filter
        start = time.perf_counter()
        pf_preds = [run_pf_sequence(z, x0_bg) for z in z_test]
        pf_time  = time.perf_counter() - start
        pf_rmse  = math.sqrt(np.mean((np.stack(pf_preds) - x_test) ** 2))

        scenario_rows.append((label, kn_rmse, kn_time, pf_rmse, pf_time))

    df_scen = pd.DataFrame(
        scenario_rows,
        columns=["Scenario", "KalmanNet_RMSE", "KalmanNet_s", "Particle_RMSE", "Particle_s"]
    )
    df_scen.to_csv(OUTPUT_DIR / "scenario_comparison.csv", index=False)

    print("\n=== Scenario comparison ===")
    print(
        df_scen.to_string(
            index=False,
            formatters={
                "KalmanNet_RMSE": "{:.4f}".format,
                "KalmanNet_s":    "{:.2f}".format,
                "Particle_RMSE":  "{:.4f}".format,
                "Particle_s":     "{:.2f}".format,
            }
        )
    )

if __name__=="__main__":
    main()