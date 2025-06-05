import numpy as np
import torch
import torch.nn as nn

# 1) Standard Linear‐Gaussian Kalman Filter
class KalmanFilter:

    def __init__(self, F, Q, H, R, x0, P0):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        # Ensure x0 is a column vector
        self.x = x0.reshape(-1, 1).astype(float)
        self.P = P0.copy().astype(float)

        self.m = self.x.shape[0]
        self.k = 0  # time index

    def _get_tvar(self, arr):
        if isinstance(arr, list):
            idx = min(self.k, len(arr) - 1)
            return arr[idx]
        else:
            return arr

    def predict(self):
        """Prediction step: x_{k|k-1} = F_k x_{k-1|k-1}, P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k."""
        Fk = self._get_tvar(self.F)
        Qk = self._get_tvar(self.Q)

        self.x = Fk @ self.x
        self.P = Fk @ self.P @ Fk.T + Qk

    def update(self, z):
        """
        Update step with measurement z:
            z: (n,) or (n×1) np.ndarray
        Returns:
            None (internal state x, P are updated)
        """
        zk = z.reshape(-1, 1).astype(float)
        Hk = self._get_tvar(self.H)
        Rk = self._get_tvar(self.R)

        # Innovation
        y = zk - (Hk @ self.x)

        # Innovation covariance
        S = Hk @ self.P @ Hk.T + Rk

        # Kalman gain
        K = self.P @ Hk.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.m)
        self.P = (I - K @ Hk) @ self.P

    def step(self, z):
        """
        One Kalman filter cycle (predict + update).
        Args:
            z : (n,) or (n×1) np.ndarray measurement
        Returns:
            x_est : (m×1) np.ndarray updated state estimate
            P_est : (m×m) np.ndarray updated covariance
        """
        self.predict()
        self.update(z)
        self.k += 1
        return self.x.copy(), self.P.copy()



class ParticleFilter:
    """
    A generic bootstrap (sequential‐importance‐resampling) particle filter.

    You must supply:
      • f_func(x_prev, w)    : state transition function returning next‐state
      • h_func(x, v)         : measurement function returning predicted measurement
      • sample_process_noise(): draws a sample w ~ N(0, Q) or according to your model
      • sample_meas_noise():    draws a sample v ~ N(0, R) or according to your model
      • meas_likelihood(z, x): returns p(z | x) (likelihood of measurement z given particle x)

    Example usage:
        pf = ParticleFilter(N=500, dim_state=3, f_func=..., h_func=...,
                            sample_process_noise=..., sample_meas_noise=...,
                            meas_likelihood=...)
        for z in measurements:
            x_est, P_est = pf.step(z)
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
        """
        Args:
            N                    : Number of particles
            dim_state            : Dimension of the state vector (m)
            f_func               : Function(x_prev, w_noise) -> x_pred
            h_func               : Function(x, v_noise) -> z_pred
            sample_process_noise : Function() -> sample w
            sample_meas_noise    : Function() -> sample v
            meas_likelihood      : Function(z, x) -> p(z | x)
            init_particles (opt) : (N×m) np.ndarray of initial particles
            init_weights  (opt) : (N,)    np.ndarray of initial normalized weights
        """
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
            # If no initial particles provided, sample from a broad Gaussian
            self.particles = np.random.randn(self.N, self.m)

        # Initialize weights
        if init_weights is not None:
            w = init_weights.astype(float)
            w /= w.sum()
            self.weights = w
        else:
            self.weights = np.ones(self.N) / self.N

    def predict(self):
        """
        Propagate each particle through the state transition with process noise.
        x_i(k) = f_func(x_i(k-1), w_i)
        """
        new_particles = np.zeros_like(self.particles)
        for i in range(self.N):
            w_i = self.sample_process_noise()   # shape (m,) or scalar
            new_particles[i] = self.f_func(self.particles[i], w_i).reshape(self.m)
        self.particles = new_particles

    def update(self, z):
        """
        Re‐weight each particle by the likelihood of measuring z given x_i.
        Then normalize weights and resample if the effective sample size is low.
        
        Args:
            z: (n,) or scalar measurement
        """
        # Compute unnormalized weights: w_i <- w_i * p(z | x_i)
        for i in range(self.N):
            likelihood = self.meas_likelihood(z, self.particles[i])
            self.weights[i] *= likelihood

        # Normalize weights
        w_sum = np.sum(self.weights)
        if w_sum == 0 or np.isnan(w_sum):
            # If all weights zero or NaN, reset to uniform
            self.weights.fill(1.0 / self.N)
        else:
            self.weights /= w_sum

        # Compute effective sample size (ESS)
        ess = 1.0 / np.sum(self.weights**2)

        # Resample if ESS is below threshold, e.g. N/2
        if ess < self.N / 2:
            self._resample()

    def _resample(self):
        """
        Systematic resampling implementation.
        """
        cumulative_sum = np.cumsum(self.weights)
        positions = (np.arange(self.N) + np.random.rand()) / self.N
        new_particles = np.zeros_like(self.particles)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        self.particles = new_particles
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """
        Return the weighted mean and covariance of the particles.
        
        Returns:
            mean (m×1), cov (m×m)
        """
        mean = np.average(self.particles, axis=0, weights=self.weights)
        diff = self.particles - mean
        cov = (diff.T * self.weights) @ diff
        return mean.reshape(self.m, 1), cov

    def step(self, z):
        """
        One PF iteration: predict(), update(z), then estimate posterior.
        
        Args:
            z: measurement at time k (shape (n,) or scalar)
        Returns:
            x_est (m×1), P_est (m×m)
        """
        self.predict()
        self.update(z)
        return self.estimate()



class KalmanNetFilter:
    """
    A hybrid Kalman filter that uses a small PyTorch network to estimate the
    Kalman gain at each time step. 

    You must supply:
      •  F:  (m×m) or list of (m×m)   state‐transition matrix
      •  Q:  (m×m) or list of (m×m)   process‐noise covariance
      •  H:  (n×m) or list of (n×m)   observation matrix
      •  R:  (n×n) or list of (n×n)   measurement‐noise covariance
      •  net: a PyTorch nn.Module that maps
            [innovation (n×1) ; P_pred.flatten() (m²×1)] → K_flat (m⋅n)
        i.e. input dimension = (n + m²), output dimension = (m⋅n)
      •  x0: (m×1)   initial state estimate
      •  P0: (m×m)   initial covariance
      •  device: "cpu" or "cuda"
    """

    def __init__(self, F, Q, H, R, net: nn.Module, x0, P0, device="cpu"):
        self.device = device
        # Convert to torch tensors as needed
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        self.net = net.to(self.device)

        # Ensure x0, P0 are torch tensors on the specified device
        self.x = torch.tensor(x0, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.P = torch.tensor(P0, dtype=torch.float32, device=self.device)

        self.m = self.x.size(0)  # state dimension
        self.k = 0               # time index

    def _get_tvar(self, arr):
        """
        Handle time‐varying versus time‐invariant arrays.
        """
        if isinstance(arr, list):
            idx = min(self.k, len(arr) - 1)
            return torch.tensor(arr[idx], dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def predict(self):
        """
        Prediction step in KalmanNet (same as standard KF predict):
            x_pred = F_k * x_prev
            P_pred = F_k * P_prev * F_k^T + Q_k
        """
        Fk = self._get_tvar(self.F)
        Qk = self._get_tvar(self.Q)

        self.x = Fk @ self.x
        self.P = Fk @ self.P @ Fk.T + Qk

    def update(self, z_numpy):
        """
        Update step using the neural net to compute Kalman gain:
        1) Innovation y = z - H_k * x_pred
        2) Innovation covariance S = H_k P_pred H_k^T + R_k
        3) K = net([y; P_pred.flatten()])
        4) x_new = x_pred + K @ y
           P_new = (I - K H_k) P_pred

        Args:
            z_numpy: (n,) or (n×1) numpy array measurement
        """
        # Convert measurement to torch
        zk = torch.tensor(z_numpy.reshape(-1, 1), dtype=torch.float32, device=self.device)

        Hk = self._get_tvar(self.H)
        Rk = self._get_tvar(self.R)

        # 1) Innovation
        y = zk - (Hk @ self.x)

        # 2) Innovation covariance (not directly used by net, but could be for diagnostics)
        S = Hk @ self.P @ Hk.T + Rk

        # 3) Prepare net input: concatenate y and P.flatten()
        P_flat = self.P.reshape(-1, 1)                       # (m²×1)
        net_input = torch.cat([y, P_flat], dim=0).T          # (1, n + m²)

        K_flat = self.net(net_input)                         # (1, m⋅n)
        K = K_flat.reshape(self.m, -1)                       # (m, n)

        # 4) State & covariance update
        self.x = self.x + K @ y
        I = torch.eye(self.m, device=self.device)
        self.P = (I - K @ Hk) @ self.P

    def step(self, z):
        """
        One KalmanNet step: predict() → update(z).
        Returns updated state and covariance as numpy arrays.

        Args:
            z: (n,) or (n×1) numpy array measurement
        """
        self.predict()
        self.update(z)
        self.k += 1
        return self.x.detach().cpu().numpy(), self.P.detach().cpu().numpy()
