import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────
# 1) Standard Linear‐Gaussian Kalman Filter
# ─────────────────────────────────────────────────────────────────────────
class KalmanFilter:
    def __init__(self, F, Q, H, R, x0, P0):
        # Store system matrices
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        # Ensure x0 is a column vector
        self.x = x0.reshape(-1, 1).astype(float)
        self.P = P0.copy().astype(float)
        self.k = 0

    def predict(self):
        # x_pred = F * x,   P_pred = F*P*F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z may be a scalar or 1×1 array; ensure it's a column vector
        z = np.atleast_2d(z).reshape(-1, 1)
        Hk = self.H
        y = z - Hk @ self.x                     # innovation
        S = Hk @ self.P @ Hk.T + self.R         # innovation covariance
        K = self.P @ Hk.T @ np.linalg.inv(S)    # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ Hk) @ self.P

    def step(self, z):
        """
        Perform one predict‐then‐update step with measurement z.
        Returns (x_est, P_est) as NumPy arrays.
        """
        self.predict()
        self.update(z)
        self.k += 1
        return self.x.copy(), self.P.copy()


# 2) Bootstrap (Sequential‐Importance‐Resampling) Particle Filter
class ParticleFilter:
    """
    A generic particle filter. You must supply:
      • f_func(x_prev, w)    : state‐transition function (returns new state)
      • h_func(x, v)         : measurement function (maps state → measurement)
      • sample_process_noise(): draws w ∼ whatever( )  (e.g. Normal(0,Q))
      • sample_meas_noise()  : draws v ∼ whatever( )  (e.g. Normal(0,R))
      • meas_likelihood(z, x): returns p(z | x) as a scalar
    """
    def __init__(self,
                 N,
                 dim_state,
                 f_func,
                 h_func,
                 sample_process_noise,
                 sample_meas_noise,
                 meas_likelihood,
                 init_particles=None,
                 init_weights=None):
        self.N = N
        self.m = dim_state
        self.f_func = f_func
        self.h_func = h_func
        self.sample_process_noise = sample_process_noise
        self.sample_meas_noise = sample_meas_noise
        self.meas_likelihood = meas_likelihood

        # Initialize particles
        if init_particles is not None:
            p = init_particles.reshape(self.N, self.m).astype(float)
            assert p.shape == (self.N, self.m)
            self.particles = p.copy()
        else:
            # Default: sample from standard normal
            self.particles = np.random.randn(self.N, self.m)

        # Initialize weights
        if init_weights is not None:
            w = init_weights.astype(float)
            w /= w.sum()
            self.weights = w.copy()
        else:
            self.weights = np.ones(self.N) / self.N

    def predict(self):
        # Propagate each particle through f_func + process noise
        for i in range(self.N):
            w = self.sample_process_noise()
            self.particles[i, :] = self.f_func(self.particles[i, :], w)

    def update(self, z):
        # Weight update: w_i ∝ w_i * p(z | x_i)
        weights = np.zeros(self.N)
        for i in range(self.N):
            weights[i] = self.weights[i] * self.meas_likelihood(z, self.particles[i, :])
        weights += 1e-300    # avoid zeros
        weights /= weights.sum()
        self.weights = weights.copy()

        # Resample if effective sample size is too low
        neff = 1.0 / np.sum(self.weights**2)
        if neff < self.N / 2.0:
            cumsum = np.cumsum(self.weights)
            step = 1.0 / self.N
            r = np.random.uniform(0, step)
            indexes = np.zeros(self.N, dtype=int)
            i, cum = 0, cumsum[0]
            for m in range(self.N):
                U = r + m * step
                while U > cumsum[i]:
                    i += 1
                indexes[m] = i
            self.particles = self.particles[indexes, :].copy()
            self.weights[:] = 1.0 / self.N

    def estimate(self):
        # Compute weighted mean & covariance of particles
        x_est = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.zeros((self.m, self.m))
        for i in range(self.N):
            diff = (self.particles[i, :] - x_est).reshape(self.m, 1)
            cov += self.weights[i] * (diff @ diff.T)
        return x_est.reshape(self.m, 1), cov

    def step(self, z):
        """
        One ParticleFilter step:
          1) predict()
          2) update(z)
          3) estimate()  → returns (x_est, P_est)
        """
        self.predict()
        self.update(z)
        return self.estimate()


# ─────────────────────────────────────────────────────────────────────────
# 3) KalmanNetFilter (a Kalman + small NN hybrid)
# ─────────────────────────────────────────────────────────────────────────
class KalmanNetFilter:
    """
    A hybrid Kalman filter that uses a neural network (PyTorch) to compute
    the Kalman gain. You must supply:
      • F : (m×m) or list of (m×m) → state‐transition matrix
      • Q : (m×m) or list of (m×m) → process‐noise covariance
      • H : (n×m) or list of (n×m) → observation matrix
      • R : (n×n) or list of (n×n) → measurement‐noise covariance
      • net: a torch.nn.Module that takes [y; P.flatten()] → returns K (m×n)
      • x0, P0: initial state & covariance
    """
    def __init__(self, F, Q, H, R, net: nn.Module, x0, P0, device="cpu"):
        self.device = device

        # Wrap F, Q, H, R as lists if they are not time‐varying
        self.F = F if isinstance(F, list) else [F]
        self.Q = Q if isinstance(Q, list) else [Q]
        self.H = H if isinstance(H, list) else [H]
        self.R = R if isinstance(R, list) else [R]
        self.net = net.to(self.device)

        self.m = x0.reshape(-1, 1).shape[0]
        self.n = self.H[0].shape[0]

        self.x = torch.tensor(x0.reshape(-1, 1), dtype=torch.float32, device=self.device)
        self.P = torch.tensor(P0, dtype=torch.float32, device=self.device)
        self.k = 0

    def _get_tvar(self, M):
        if isinstance(M, list):
            idx = min(self.k, len(M) - 1)      # 0 for constant matrices
            return torch.tensor(M[idx], dtype=torch.float32, device=self.device)
        return torch.tensor(M, dtype=torch.float32, device=self.device)


    def predict(self):
        # Standard KF predict step with torch tensors
        Fk = self._get_tvar(self.F)
        Qk = self._get_tvar(self.Q)
        self.x = Fk @ self.x
        self.P = Fk @ self.P @ Fk.T + Qk

    def update(self, z_numpy):
        """
        1) Compute innovation y = z - H_k x_pred
        2) Compute S = H_k P_pred H_k^T + R_k
        3) Flatten P_pred, concatenate with y, pass into net to get K
        4) x_new = x_pred + K y;  P_new = (I - K H_k) P_pred
        """
        zk = torch.tensor(z_numpy.reshape(-1, 1), dtype=torch.float32, device=self.device)
        Hk = self._get_tvar(self.H)
        Rk = self._get_tvar(self.R)

        y = zk - Hk @ self.x
        S = Hk @ self.P @ Hk.T + Rk

        P_flat = self.P.flatten().reshape(-1, 1)
        inp = torch.cat((y, P_flat), dim=0)

        K_vec = self.net(inp.T)
        K = K_vec.view(self.m, self.n)

        self.x = self.x + K @ y
        I = torch.eye(self.P.shape[0], device=self.device)
        self.P = (I - K @ Hk) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        self.k += 1
        return self.x.detach().cpu().numpy(), self.P.detach().cpu().numpy()
