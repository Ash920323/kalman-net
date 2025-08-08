import sys
from pathlib import Path
src_folder = Path.cwd().parent     
sys.path.insert(0, str(src_folder))

from util import filter
import numpy as np
import pandas as pd
from scipy.integrate import ode
import matplotlib.pyplot as plt
from numpy.random import default_rng
# print cwd
rng = default_rng(42) #random number generator
# the parameters are converted so the units could match the reference paper
params = {
    # —— Glucose subsystem (eq. 27) —————————————————————
    "PG"      : 0.022,      # 1/min
    "Si"      : 1.2e-4,     # 1/min
    "alpha_G" : 1.0,        # 1/(mU·L⁻¹)
    "EGPb"    : 1.27,       # mmol/min  (0.0161 mmol⋅kg⁻¹⋅min⁻¹ × 79 kg)
    "CNS"     : 1.16,       # mmol/min
    "VG"      : 10.0,       # L

    # —— Insulin transport (eqs. 28:29) —————————————————
    "nI"      : 0.157,      # 1/min
    "nC"      : 0.0159,     # 1/min
    "nK"      : 0.0165,     # 1/min
    "nL"      : 0.003,      # 1/min
    "alpha_I" : 0.18,       # 1/(mU·L⁻¹)
    "VI"      : 11.0,       # L
    "xL"      : 0.60,       # unitless

    # —— Endogenous insulin (eq. 32) ——————————————————
    "k1"      : 210,        # mU/min
    "k2"      : 300,        # mU/L
    "k3"      : 0.007,      # unitless

    # —— Stomach / gut (eqs. 30–31) ————————————————
    "d1"      : 0.05,       # 1/min
    "d2"      : 0.06,       # 1/min
    "Pmax"    : 1.50,       # mmol/min
    "PN"      : 0.011,      # mmol/min  (0.2 mg/min ÷ 180 mg/mmol)

    # —— Interstitial glucose (eq. 34) ——————————————
    "beta1"   : 0.002,      # 1/min
    "beta2"   : 0.02,       # 1/min

    # —— Constant insulin infusion ——————————————————
    "uex"     : 58.91       # mU/min
}


output_dir = Path.cwd() / "run_0"
output_dir.mkdir(exist_ok=True)

def D_of_t(t_min):
    meal_starts = (7*60, 14*60, 21*60)      # the time of the day for the meal intake
    D_rate = 5 * (1000/180) / 40          # = 5.556 mmol/min
    return sum(D_rate for m in meal_starts if m <= t_min < m + 40)


def glucose_insulin_model(t, x, p):
    BG, IG, Q, I, P1, P2 = x
    P = p["PN"] + min(p["d2"] * P2, p["Pmax"])
    uen = p["k1"] * np.exp(-I / p["k2"]) + p["k3"]

    dBG = ( -p["PG"]*BG
            - p["Si"]*BG*Q/(1 + p["alpha_G"]*Q)
            + P
            + p["EGPb"]
            - p["CNS"] ) / p["VG"]

    dQ  = p["nI"]*(I - Q) - p["nC"]*Q/(1 + p["alpha_G"]*Q)

    dI  = ( -p["nK"]*I
            - p["nL"]*I/(1 + p["alpha_I"]*I)
            - p["nI"]*(I - Q)
            + p["uex"]/p["VI"]
            + (1 - p["xL"]) * uen / p["VI"] )

    dP1 = -p["d1"]*P1 + D_of_t(t)        # D_of_t defined below
    dP2 = -min(p["d2"]*P2, p["Pmax"]) + p["d1"]*P1
    dIG = p["beta1"]*BG - p["beta2"]*IG

    return [dBG, dIG, dQ, dI, dP1, dP2]



t_meal   = np.arange(0, 48*60+1)          # 48 h, 1-min grid
D_trace  = [D_of_t(t) for t in t_meal]

plt.figure(figsize=(8,1.8))
plt.step(t_meal/60, D_trace, where='post', lw=2)
plt.ylabel('$D$ (mmol/min)')
plt.xlabel('Time [h]')
plt.title('Enteral feed profile ')
plt.tight_layout()

plt.savefig(output_dir/"enteral_feed_profile.png")
plt.close()


t_end = 48 * 60        # plotting 48 hours
dt    = 1
ts    = np.arange(0, t_end + dt, dt)

# Initial state vector: [BG, IG, Q, I, P1, P2]
x0 = np.array([5.0, 5.0, 74.0, 108.58, 0.0, 0.0])
x0_nominal = np.array([5.0, 5.0, 74.0, 108.58, 0.0, 0.0])   # BG, IG, Q, I, P1, P2
perturb_pct   = [0.10, 0.20, 0.30]                          # 10 %, 20 %, 30 %
x0_list = []

for p in perturb_pct:
    for sign in (+1, -1):
        vec = x0_nominal * (1 + sign * p)
        vec = np.clip(vec, 0.0, None)   #  numbers cannot be negative
        x0_list.append(vec)

x0_labels = [f"±{int(p*100)}% {('up' if i%2==0 else 'down')}"
             for p in perturb_pct
             for i in range(2)]

# display
# display(pd.DataFrame(x0_list, index=x0_labels, columns=["BG", "IG", "Q", "I", "P1", "P2"]))

df = pd.DataFrame(x0_list, index=x0_labels,
                  columns=["BG", "IG", "Q", "I", "P1", "P2"])
df.to_csv(output_dir/"initial_states.csv")

solver = ode(lambda t, x: glucose_insulin_model(t, x, params))\
           .set_integrator('dopri5')
solver.set_initial_value(x0, 0)

xs = np.zeros((len(ts), 6))
xs[0] = x0
for k in range(1, len(ts)):
    xs[k] = solver.integrate(ts[k])

# Plot the true BG trajectory
plt.figure(figsize=(8, 3))
plt.plot(ts / 60, xs[:, 0], label='True BG')
plt.xlabel('Time [h]')
plt.ylabel('BG [mM]')
plt.title('True blood-glucose (no noise)')
plt.tight_layout()

plt.savefig(output_dir/"true_bg_trajectory.png")
plt.close()

# glucouse level data with normal noise
rng = np.random.default_rng(seed=42)    # for reproducibility
scale_gauss = 0.2                       # standard deviation of Gaussian noise
bg_true = xs[:, 0]                      # true BG from your simulation
time_hours = ts / 60                    # convert time vector to hours

# generate Gaussian noise and add it to the true BG
noise_gauss = rng.normal(loc=0.0,
                         scale=scale_gauss,
                         size=bg_true.shape)
bg_gauss_noisy = bg_true + noise_gauss

plt.figure(figsize=(8, 3))
plt.plot(time_hours, bg_true,          label='True BG',        color='tab:blue')
plt.plot(time_hours, bg_gauss_noisy,   label='Gaussian noise',  color='tab:orange', alpha=0.6)
plt.xlabel('Time [h]')
plt.ylabel('BG [mM]')
plt.title(f'Blood‐glucose with Gaussian Noise (σ = {scale_gauss})')
plt.legend()
plt.tight_layout()

plt.savefig(output_dir/"bg_with_gaussian_noise.png")
plt.close()

scale_laplace = 0.2  # scale parameter for Laplace (equivalent to σ for comparison)
noise_laplace = rng.laplace(loc=0.0, scale=scale_laplace, size=bg_true.shape)
bg_laplace_noisy = bg_true + noise_laplace

plt.figure(figsize=(8, 3))
plt.plot(time_hours, bg_true, label='True BG', color='tab:blue')
plt.plot(time_hours, bg_gauss_noisy, label='Gaussian noise', color='tab:orange', alpha=0.6)
plt.plot(time_hours, bg_laplace_noisy, label='Laplace noise', color='tab:green', alpha=0.6)
plt.xlabel('Time [h]')
plt.ylabel('BG [mM]')
plt.title('Blood-glucose with Laplace Noise (scale = 0.2)')
plt.legend()
plt.tight_layout()

plt.savefig(output_dir/"bg_with_laplace_noise.png")
plt.close()



import os


print("Current working directory:", os.getcwd())

# Change directory to the src folder
SRC_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))  # project_root/src
print("Changing directory to:", SRC_PATH)
os.chdir(SRC_PATH)

# Import the KalmanFilter class from util.filter
from util.filter import KalmanFilter, KalmanNetFilter, ParticleFilter
print(KalmanFilter)
print(KalmanNetFilter)
print(ParticleFilter)


# ------------------------------------------------------------------------
# Particle-filter denoising of the CGM traces
# ---------------------------------------------------------------------------
from util.filter import ParticleFilter

# 1.  Model definition -------------------------------------------------------
dim_state   = 1                 # BG is a single scalar state
Q_STD       = 0.05              # process-noise σ  (tune to allow more / less drift)
SIGMA_G     = 0.2               # measurement σ for the Gaussian-noise trace
B_LAPLACE   = scale_laplace     # Laplace “b” parameter (= σ/√2)

def f_func(x_prev, w):
    """Random-walk process model:  Gₖ = Gₖ₋₁ + wₖ"""
    return x_prev + w           # x_prev and w are shape-(1,) arrays

def h_func(x, v):
    """Ideal CGM sensor: zₖ = Gₖ + vₖ"""
    return x + v

# Sampling helpers -----------------------------------------------------------
sample_process_noise = lambda: np.random.normal(0.0, Q_STD, (dim_state,))
sample_meas_noise_g  = lambda: np.random.normal(0.0, SIGMA_G, (dim_state,))
sample_meas_noise_l  = lambda: np.random.laplace(0.0, B_LAPLACE, (dim_state,))

# Likelihood functions -------------------------------------------------------
norm_coef     = 1.0 / (np.sqrt(2 * np.pi) * SIGMA_G)
laplace_coef  = 1.0 / (2.0 * B_LAPLACE)
"""
def meas_like_gauss(z, x):
    return norm_coef * np.exp(-0.5 * ((z - x[0]) / SIGMA_G) ** 2)
"""
def meas_like_laplace(z, x):
    """p(z | x) under Laplace(x, b)"""
    return laplace_coef * np.exp(-np.abs(z - x[0]) / B_LAPLACE)

# 2.  Instantiate one PF for each noise type ---------------------------------
N_PARTICLES  = 5_00
INIT_SPREAD  = 0.5              # stddev of initial guess (mM)
"""
pf_gauss = ParticleFilter(
    N=N_PARTICLES,
    dim_state=dim_state,
    f_func=f_func,
    h_func=h_func,
    sample_process_noise=sample_process_noise,
    sample_meas_noise=sample_meas_noise_g,
    meas_likelihood=meas_like_gauss,
    init_particles=np.random.normal(bg_gauss_noisy[0],
                                    INIT_SPREAD,
                                    size=(N_PARTICLES, dim_state)),
)
"""
pf_laplace = ParticleFilter(
    N=N_PARTICLES,
    dim_state=dim_state,
    f_func=f_func,
    h_func=h_func,
    sample_process_noise=sample_process_noise,
    sample_meas_noise=sample_meas_noise_l,
    meas_likelihood=meas_like_laplace,
    init_particles=np.random.normal(bg_laplace_noisy[0],
                                    INIT_SPREAD,
                                    size=(N_PARTICLES, dim_state)),
)

# 3.  Run the two filters -----------------------------------------------------
#est_gauss   = np.empty_like(bg_gauss_noisy)
est_laplace = np.empty_like(bg_laplace_noisy)

"""for k, z in enumerate(bg_gauss_noisy):
    x_hat, _  = pf_gauss.step(z)
    est_gauss[k] = x_hat"""

for k, z in enumerate(bg_laplace_noisy):
    x_hat, _   = pf_laplace.step(z)
    est_laplace[k] = x_hat

# 4.  Optional RMSE sanity check ---------------------------------------------
#rmse_gauss   = np.sqrt(np.mean((est_gauss   - bg_true) ** 2))
rmse_laplace = np.sqrt(np.mean((est_laplace - bg_true) ** 2))
#print(f"Particle-filter RMSE : Gaussian noise : {rmse_gauss:0.3f} mM")
print(f"Particle-filter RMSE : Laplace noise  : {rmse_laplace:0.3f} mM")

# 5.  Visual comparison -------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_hours, bg_true,          label='True BG')
plt.plot(time_hours, bg_laplace_noisy, '.', alpha=0.3, label='Noisy (Laplace)')
plt.plot(time_hours, est_laplace,      label='PF estimate')
plt.xlabel('Time [h]'); plt.ylabel('BG [mM]')
plt.title('Particle-filter denoising : Laplace-corrupted trace')
plt.legend(); plt.tight_layout();

plt.savefig(output_dir/"pf_denoising_laplace.png")
plt.close()


# ---------------------------------------------------------------------------
# Kalman-filter denoising of the CGM traces
# ---------------------------------------------------------------------------
from util.filter import KalmanFilter

# 1.  Model matrices for a *random-walk* state model  Gk = Gk-1 + wk
F = np.array([[1.0]])                 # state-transition  (identity)
H = np.array([[1.0]])                 # observation       (direct read-out)

Q = np.array([[Q_STD**2]])            # process-noise variance (scalar)
R_gauss   = np.array([[SIGMA_G**2]])  # measurement variance for Gaussian sensor
R_laplace = np.array([[2*B_LAPLACE**2]])
# Laplace(b) has variance 2 b², so we approximate it with an equivalent Gaussian
# in order to keep the KF mathematically valid.

# 2.  Initial priors (start at first reading, with conservative uncertainty)
x0_gauss = np.array([bg_gauss_noisy[0]])
P0_gauss = np.array([[SIGMA_G**2]])

x0_lap   = np.array([bg_laplace_noisy[0]])
P0_lap   = np.array([[2*B_LAPLACE**2]])

kf_gauss = KalmanFilter(F, Q, H, R_gauss,   x0_gauss, P0_gauss)
kf_lap   = KalmanFilter(F, Q, H, R_laplace, x0_lap,   P0_lap)

# 3.  Run the two filters
est_gauss_kf   = np.empty_like(bg_gauss_noisy)
est_laplace_kf = np.empty_like(bg_laplace_noisy)

for k, z in enumerate(bg_gauss_noisy):
    x_hat, _ = kf_gauss.step(z)          # predict → update
    est_gauss_kf[k] = x_hat.squeeze()    # store scalar

for k, z in enumerate(bg_laplace_noisy):
    x_hat, _ = kf_lap.step(z)
    est_laplace_kf[k] = x_hat.squeeze()

# 4.  Quality check
rmse_gauss_kf   = np.sqrt(np.mean((est_gauss_kf   - bg_true)**2))
rmse_laplace_kf = np.sqrt(np.mean((est_laplace_kf - bg_true)**2))
print(f"Kalman-filter RMSE : Gaussian noise : {rmse_gauss_kf:0.3f} mM")
print(f"Kalman-filter RMSE : Laplace noise  : {rmse_laplace_kf:0.3f} mM")

# 5.  Visual comparison
plt.figure(figsize=(9,4))
plt.plot(time_hours, bg_true,        label='True BG')
plt.plot(time_hours, bg_gauss_noisy, '.', alpha=0.30, label='Noisy (Gaussian)')
plt.plot(time_hours, est_gauss_kf,   label='KF estimate')
plt.xlabel('Time [h]'); plt.ylabel('BG [mM]')
plt.title('Kalman-filter denoising : Gaussian-corrupted trace')
plt.legend(); plt.tight_layout()

plt.savefig(output_dir/"kf_denoising_gaussian.png")
plt.close()

plt.figure(figsize=(9,4))
plt.plot(time_hours, bg_true,          label='True BG')
plt.plot(time_hours, bg_laplace_noisy, '.', alpha=0.30, label='Noisy (Laplace)')
plt.plot(time_hours, est_laplace_kf,    label='KF estimate')
plt.xlabel('Time [h]'); plt.ylabel('BG [mM]')
plt.title('Kalman-filter denoising : Laplace-corrupted trace')
plt.legend(); plt.tight_layout()

plt.savefig(output_dir/"kf_denoising_laplace.png")
plt.close()


def sample_meal_times(rng=None,
                      first_window=(7 * 60, 11 * 60),   # 07:00 – 11:00
                      second_window=(12 * 60, 14 * 60), # 12:00 – 14:00
                      third_window=(18 * 60, 21 * 60),  # 18:00 – 21:00
                      min_gap=5 * 60):                  # ≥ 5 h between meals
    """
    Draw three meal–start times (in minutes after midnight) that
    satisfy the user’s constraints.
    """

    while True:
        t1 = rng.integers(*first_window)
        t2 = rng.integers(*second_window)
        t3 = rng.integers(*third_window)

        if (t2 - t1) >= min_gap and (t3 - t2) >= min_gap:
            return int(t1), int(t2), int(t3)


def make_D_of_t(meal_times,
                D_rate=5 * (1000 / 180) / 40,  # mmol · min⁻¹
                meal_duration=40):
    """
    Return a function D_of_t(t_min) that is piece-wise constant.
    """
    meal_times = np.asarray(meal_times)

    def D_of_t(t_min):
        # works for scalar or NumPy array input
        t = np.asarray(t_min)
        active = ((t[:, None] if t.ndim else t) >= meal_times) & (
            (t[:, None] if t.ndim else t) < (meal_times + meal_duration))
        return D_rate * active.sum(axis=-1)

    return D_of_t


# ------------------------------------------------------------------
# Demo: show three random scenarios in a single day
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
scenarios = [sample_meal_times(rng) for _ in range(3)]

t = np.arange(24 * 60)  # 0 … 1440 min (one day)

plt.figure(figsize=(9, 4))
for mt in scenarios:
    D_func = make_D_of_t(mt)
    D_trace = D_func(t)
    label = f"{mt[0]//60:02d}:{mt[0]%60:02d}, " \
            f"{mt[1]//60:02d}:{mt[1]%60:02d}, " \
            f"{mt[2]//60:02d}:{mt[2]%60:02d}"
    plt.plot(t / 60, D_trace, label=label)

plt.xlabel("Time of day (h)")
plt.ylabel("D  (mmol / min)")
plt.title("Random meal schedules satisfying ≥ 5 h gaps")
plt.legend(title="Meal starts")
plt.tight_layout()


import numpy as np
from scipy.integrate import ode
import torch
from pathlib import Path
from tqdm.auto import tqdm

# ---------------- configuration --------------------------------------------------
rng              = np.random.default_rng(seed=42)
N_TRAIN_STANDARD = 200
N_TRAIN_MIXED    = 200
N_TEST           = 200
B_LAPLACE        = 0.20
STANDARD_MEALS   = 3
t_end            = 48 * 60
dt               = 1
ts               = np.arange(0, t_end + dt, dt)
T                = ts.size

# ------------------------------------------------------------------------------
# 1) figure out where we're writing
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()
if base_dir.name == "notebooks":
    base_dir = base_dir.parent
if base_dir.name != "src":
    base_dir = base_dir / "src"

out_dir = base_dir / "data_knet"
out_dir.mkdir(parents=True, exist_ok=True)

# 2) paths to the 3 files we expect
files = {
    "train_standard": out_dir / "train_standard_laplace.npz",
    "train_mixed":   out_dir / "train_mixed_laplace.npz",
    "test":          out_dir / "test_laplace.npz",
}

# ------------------------------------------------------------------------------
# 3) load if they all already exist
if all(p.exists() for p in files.values()):
    print("✅ Found existing datasets, loading…")
    train_std_npz = np.load(files["train_standard"])
    z_train_std, train_std_true, meal_train_std = (
        train_std_npz["z"],
        train_std_npz["x"],
        train_std_npz["meal"],
    )
    train_mix_npz = np.load(files["train_mixed"])
    z_train_mix, train_mix_true, meal_train_mix = (
        train_mix_npz["z"],
        train_mix_npz["x"],
        train_mix_npz["meal"],
    )
    test_npz = np.load(files["test"])
    z_test, test_true, meal_test = (
        test_npz["z"],
        test_npz["x"],
        test_npz["meal"],
    )

# 4) otherwise simulate & save
else:
    print("⚙️  Datasets not found, running simulation…")

    BASE_RATE = 5 * (1000 / 180) / 40
    def sample_meal_times(rng, n_meals=STANDARD_MEALS):
        latest = t_end - 60
        return np.sort(rng.choice(latest, size=n_meals, replace=False))

    def simulate_bg_trace(meal_times, total_meals=STANDARD_MEALS):
        scale = total_meals / len(meal_times)
        def D_of_t_local(t):
            return sum(BASE_RATE * scale
                       for m in meal_times
                       if m <= t < m + 40)
        globals()["D_of_t"] = D_of_t_local

        solver = ode(lambda t, x: glucose_insulin_model(t, x, params))
        solver.set_integrator("dopri5")
        solver.set_initial_value(x0, 0.0)

        xs = np.zeros((T, 6), np.float32)
        xs[0] = x0
        for k in range(1, T):
            xs[k] = solver.integrate(ts[k])
        return xs[:, 0]

    def make_split(N, desc, mixed=False):
        max_meals = 5 if mixed else STANDARD_MEALS
        bg_true = np.zeros((N, T), np.float32)
        z_lap   = np.zeros_like(bg_true)
        meals_padded = -np.ones((N, max_meals), np.int16)

        for i in tqdm(range(N), desc=desc):
            nm = rng.integers(3, 6) if mixed else STANDARD_MEALS
            mt = sample_meal_times(rng, nm)
            meals_padded[i, :nm] = mt
            bg_true[i] = simulate_bg_trace(mt)
            z_lap[i]   = bg_true[i] + rng.laplace(0.0, B_LAPLACE, bg_true[i].shape)

        return bg_true, z_lap, meals_padded

    train_std_true, z_train_std, meal_train_std = make_split(N_TRAIN_STANDARD, "train_standard")
    train_mix_true, z_train_mix, meal_train_mix = make_split(N_TRAIN_MIXED,    "train_mixed", mixed=True)
    test_true,      z_test,      meal_test      = make_split(N_TEST,           "test_standard")

    # save
    np.savez_compressed(files["train_standard"], z=z_train_std, x=train_std_true, meal=meal_train_std)
    np.savez_compressed(files["train_mixed"],    z=z_train_mix,  x=train_mix_true,  meal=meal_train_mix)
    np.savez_compressed(files["test"],           z=z_test,       x=test_true,       meal=meal_test)
    print(f"✅  Datasets simulated and saved to {out_dir}")


class GlucoseDataset(torch.utils.data.Dataset):
    def __init__(self, z, x_true):
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.x = torch.as_tensor(x_true, dtype=torch.float32)
    def __len__(self): return len(self.z)
    def __getitem__(self, idx): return self.z[idx], self.x[idx]

train_std_dl = torch.utils.data.DataLoader(GlucoseDataset(z_train_std, train_std_true), batch_size=32, shuffle=True,  drop_last=True)
train_mix_dl = torch.utils.data.DataLoader(GlucoseDataset(z_train_mix, train_mix_true), batch_size=32, shuffle=True,  drop_last=True)
test_dl      = torch.utils.data.DataLoader(GlucoseDataset(z_test,      test_true),      batch_size=32, shuffle=False, drop_last=True)

print("train_std batches:", len(train_std_dl))
print("train_mix batches:", len(train_mix_dl))
print("test batches     :", len(test_dl))


def load_dataset_files(data_dir: Path):
    """Load train‑standard / train‑mixed / test sets from `*.npz` files."""

    def _load(fname: str):
        arr = np.load(data_dir / fname)
        return arr["z"], arr["x"]

    z_train_std, x_train_std = _load("train_standard_laplace.npz")
    z_train_mix, x_train_mix = _load("train_mixed_laplace.npz")
    z_test,      x_test      = _load("test_laplace.npz")

    return (GlucoseDataset(z_train_std, x_train_std),
            GlucoseDataset(z_train_mix, x_train_mix),
            GlucoseDataset(z_test,      x_test     ))


import itertools, json, time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)
np.random.seed(42)

# Adjust these to taste
DATA_DIR  = Path.cwd() / "kalman-net" / "src" / "data_knet"   # folder with the *.npz files
RESULTS_DIR = Path.cwd() / "kalman-net" / "src" / "results"   # where JSON + plots will be saved
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRAC   = 0.10        # share of training data used for validation
BATCH_SZ   = 256         # data‑loader batch size
MAX_EPOCHS = 5          # upper bound for epoch‑count tuning

train_std_ds, train_mix_ds, test_ds = load_dataset_files(DATA_DIR)
print("Datasets loaded ✔️")


class GainGRU(nn.Module):
    """Minimal GRU→Linear(σ) network predicting the Kalman gain per step."""
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
    def forward(self, z_seq):          # z_seq: (B, T, 1)
        out, _ = self.gru(z_seq)
        return self.fc(out).squeeze(-1) # (B, T)

F   = torch.tensor([[1.0]],  dtype=torch.float32, device=DEVICE)
Q   = torch.tensor([[0.01]], dtype=torch.float32, device=DEVICE)
H   = torch.tensor([[1.0]],  dtype=torch.float32, device=DEVICE)
EYE = torch.eye(1, device=DEVICE)
MSE = nn.MSELoss()

def run_filter_through_net(net: nn.Module, z_seq: torch.Tensor):
    """Rollout a 1‑D Kalman filter using NN‑predicted gains."""
    K_seq = net(z_seq).squeeze(0)
    x_hat = z_seq[:,0].view(1,1)
    P     = torch.tensor([[5.0]], device=DEVICE)
    est   = []
    for t in range(z_seq.shape[1]):
        x_pred = F @ x_hat
        P_pred = F @ P @ F.T + Q
        z_t    = z_seq[:,t].view(1,1)
        K_t    = K_seq[t].view(1,1)
        x_hat  = x_pred + K_t @ (z_t - H @ x_pred)
        P      = (EYE - K_t @ H) @ P_pred
        est.append(x_hat.view(-1))
    return torch.stack(est).squeeze()

PARAM_GRID = {
    "hidden_size": [16],
    "num_layers" : [1, 2],
    "lr"         : [3e-4, 1e-4],
    "dropout"    : [0.0],
    "grad_clip"  : [0.5, 1.0],
    "epochs"     : [4],   # short while searching
}
print(f"Grid size = {np.prod([len(v) for v in PARAM_GRID.values()])} trials")


def build_loaders(ds: GlucoseDataset):
    val_sz = int(len(ds)*VAL_FRAC)
    train_sz = len(ds) - val_sz
    train_ds, val_ds = random_split(ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SZ, shuffle=False, drop_last=False)
    return train_dl, val_dl

def batch_loss(net, z_b, x_b, grad_clip=None, opt=None):
    B, T = z_b.shape; loss = 0.0
    for b in range(B):
        est = run_filter_through_net(net, z_b[b].view(1,T,1))
        loss += MSE(est, x_b[b])
    loss /= B
    if opt is not None:
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()
    return loss.item()

train_dl_search, val_dl_search = build_loaders(train_std_ds)

search_space = list(itertools.product(*PARAM_GRID.values()))
results = []
start = time.perf_counter()
for i, combo in enumerate(search_space, 1):
    hp = dict(zip(PARAM_GRID.keys(), combo))
    net = GainGRU(hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    for _ in range(hp["epochs"]):
        net.train()
        for z_b, x_b in train_dl_search:
            batch_loss(net, z_b.to(DEVICE), x_b.to(DEVICE), hp["grad_clip"], opt)
    # validation
    net.eval(); val_mse = 0.0
    with torch.no_grad():
        for z_b, x_b in val_dl_search:
            val_mse += batch_loss(net, z_b.to(DEVICE), x_b.to(DEVICE))
    val_mse /= len(val_dl_search)
    results.append((val_mse, hp))
    print(f"Trial {i:02d}/{len(search_space)} | val MSE = {val_mse:.4f} | {hp}")

best_val, best_hp = min(results, key=lambda t: t[0])
print("\n🏆 Best grid‑search config:", best_hp, "| val MSE =", f"{best_val:.4f}")
json.dump({"results": results, "best": best_hp}, open(RESULTS_DIR/"grid_search_results.json","w"), indent=2)
print("Grid‑search JSON saved ✔️")

train_dl_tune, val_dl_tune = build_loaders(train_std_ds)

hp = {k:v for k,v in best_hp.items() if k!="epochs"}
net = GainGRU(hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(DEVICE)
opt = optim.Adam(net.parameters(), lr=hp["lr"])

val_curve = []
best_epoch, best_val = 0, float("inf")
for ep in range(1, MAX_EPOCHS+1):
    net.train()
    for z_b, x_b in train_dl_tune:
        batch_loss(net, z_b.to(DEVICE), x_b.to(DEVICE), hp["grad_clip"], opt)
    # validation
    net.eval(); val_mse = 0.0
    with torch.no_grad():
        for z_b, x_b in val_dl_tune:
            val_mse += batch_loss(net, z_b.to(DEVICE), x_b.to(DEVICE))
    val_mse /= len(val_dl_tune)
    val_curve.append(val_mse)
    if val_mse < best_val:
        best_val, best_epoch = val_mse, ep
    print(f"Epoch {ep:02d} — val MSE: {val_mse:.4f}")

plt.figure(figsize=(6,3))
plt.plot(range(1, MAX_EPOCHS+1), val_curve, marker='o')
plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.title("Epoch Tuning")
plt.grid(True); plt.tight_layout()
plt.savefig(RESULTS_DIR/"epoch_tuning_curve.png")
plt.savefig(RESULTS_DIR/"epoch_tuning_curve.png")
plt.close()

json.dump({"best_epoch": best_epoch}, open(RESULTS_DIR/"best_epoch.json","w"))
print(f"🏁 Best #epochs = {best_epoch} | val MSE = {best_val:.4f}")

hp_epochs = hp | {"epochs": best_epoch}
loaders = {
    "standard": DataLoader(train_std_ds, batch_size=BATCH_SZ, shuffle=True, drop_last=False),
    "mixed":    DataLoader(train_mix_ds, batch_size=BATCH_SZ, shuffle=True, drop_last=False)
}
final_models = {}
for name, dl in loaders.items():
    print(f"\nTraining {name.upper()} model …")
    net = GainGRU(hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    for _ in range(best_epoch):
        net.train()
        for z_b, x_b in dl:
            batch_loss(net, z_b.to(DEVICE), x_b.to(DEVICE), hp["grad_clip"], opt)
    final_models[name] = net
print("Final models trained ✔️")

z_test, x_test = test_ds.z.numpy(), test_ds.x.numpy()

def gru_filter(net: nn.Module, z_np: np.ndarray):
    net.eval()
    z_t = torch.tensor(z_np, dtype=torch.float32, device=DEVICE).view(1,-1,1)
    with torch.no_grad():
        est = run_filter_through_net(net, z_t)
    return est.cpu().numpy()

def evaluate(net):
    start = time.perf_counter()
    preds = [gru_filter(net, z) for z in z_test]
    inf_time = time.perf_counter()-start
    rmse = np.sqrt(((np.stack(preds)-x_test)**2).mean())
    return rmse, inf_time

print("\n============== Performance ==============")
for name, model in final_models.items():
    rmse, t_inf = evaluate(model)
    print(f"{name.title():10s} | RMSE = {rmse:.4f} | inference = {t_inf:.2f}s")
