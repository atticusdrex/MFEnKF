import numpy as np
import matplotlib.pyplot as plt

print("Initializing 1D Wave Equation Time-Series Simulation...")

N_pts = 100
state_dim = 2 * N_pts
L = 10.0
dx = L / N_pts
t_end = 5.0
dt = 0.01
n_steps = int(np.ceil(t_end / dt))
tspan = np.linspace(0, t_end, n_steps)
process_var = 1e-4
obs_var = 1e-2

obs_indices = np.arange(0, N_pts, 10)
obs_dim = len(obs_indices)
H = np.zeros((obs_dim, state_dim))
for i, idx in enumerate(obs_indices):
    H[i, idx] = 1.0

def wave_deriv(X, dx, c):
    u, v = X[:N_pts], X[N_pts:]
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / (dx**2)
    return np.concatenate([v, (c**2) * u_xx])

def hf_deriv(X): return wave_deriv(X, dx, c=1.0)  # True physics
def lf_deriv(X): return wave_deriv(X, dx, c=0.8)  # Cheap physics

def random_rk4_step(X, dXdt, dt, var):
    k1 = dXdt(X)
    k2 = dXdt(X + 0.5 * dt * k1)
    k3 = dXdt(X + 0.5 * dt * k2)
    k4 = dXdt(X + dt * k3)
    X_next = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return X_next + np.sqrt(var) * np.random.normal(size=X.shape)

def enkf_forecast(ensemble, dXdt, dt, process_var):
    state_dim, N = ensemble.shape
    ens_f = np.zeros_like(ensemble)
    for i in range(N):
        ens_f[:, i] = random_rk4_step(ensemble[:, i], dXdt, dt, process_var)
    return ens_f

def enkf_analysis(ensemble_f, y, H, obs_var):
    state_dim, N = ensemble_f.shape
    obs_dim = y.shape[0]
    x_mean = ensemble_f.mean(axis=1, keepdims=True)
    A = (ensemble_f - x_mean) / np.sqrt(N - 1)
    
    P_xy = A @ (H @ A).T
    P_yy = (H @ A) @ (H @ A).T + obs_var * np.eye(obs_dim)
    
    K = np.linalg.solve(P_yy.T, P_xy.T).T
    eps = np.sqrt(obs_var) * np.random.normal(size=(obs_dim, N))
    innovation = (y[:, None] + eps) - H @ ensemble_f
    return ensemble_f + K @ innovation

def run_enkf(X_obs, X0_true, n_ensemble):
    ens = X0_true[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim, n_ensemble))
    means, stds = np.zeros((state_dim, n_steps)), np.zeros((state_dim, n_steps))
    means[:, 0], stds[:, 0] = ens.mean(axis=1), ens.std(axis=1)

    for i in range(1, n_steps):
        ens = enkf_forecast(ens, hf_deriv, dt, process_var)
        ens = enkf_analysis(ens, X_obs[:, i], H, obs_var)
        means[:, i], stds[:, i] = ens.mean(axis=1), ens.std(axis=1)
    return means, stds

def run_mfenkf(X_obs, X0_true, hf_size, lf_size):
    hf_ens = X0_true[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim, hf_size))
    linked_ens = np.copy(hf_ens)
    lf_ens = X0_true[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim, lf_size))

    means, stds = np.zeros((state_dim, n_steps)), np.zeros((state_dim, n_steps))
    means[:, 0] = hf_ens.mean(axis=1)
    stds[:, 0] = lf_ens.std(axis=1) + (hf_ens - linked_ens).std(axis=1)

    for i in range(1, n_steps):
        hf_ens = enkf_forecast(hf_ens, hf_deriv, dt, process_var)
        linked_ens = enkf_forecast(linked_ens, lf_deriv, dt, process_var)
        lf_ens = enkf_forecast(lf_ens, lf_deriv, dt, process_var)
        
        y = X_obs[:, i]
        hf_ens = enkf_analysis(hf_ens, y, H, obs_var)
        linked_ens = enkf_analysis(linked_ens, y, H, obs_var)
        lf_ens = enkf_analysis(lf_ens, y, H, obs_var)
        
        means[:, i] = hf_ens.mean(axis=1) + (lf_ens.mean(axis=1) - linked_ens.mean(axis=1))
        stds[:, i] = lf_ens.std(axis=1) + (hf_ens - linked_ens).std(axis=1)
    return means, stds

if __name__ == "__main__":
    print("Generating ground truth and sparse observations...")
    x_space = np.linspace(0, L, N_pts)
    X0 = np.concatenate([np.exp(-((x_space - L/2)**2)), np.zeros(N_pts)])

    X_true = np.zeros((state_dim, n_steps))
    X_lf_true = np.zeros((state_dim, n_steps))
    X_obs = np.zeros((obs_dim, n_steps))
    X_true[:, 0], X_lf_true[:, 0] = X0, X0
    X_obs[:, 0] = H @ X0

    curr_X, curr_X_lf = X0, X0
    for i in range(1, n_steps):
        curr_X = random_rk4_step(curr_X, hf_deriv, dt, var=process_var)
        curr_X_lf = random_rk4_step(curr_X_lf, lf_deriv, dt, var=process_var)
        X_true[:, i], X_lf_true[:, i] = curr_X, curr_X_lf
        X_obs[:, i] = H @ curr_X + np.sqrt(obs_var) * np.random.normal(size=(obs_dim,))

    print("Running Standard EnKF (2 members)...")
    means_enkf, stds_enkf = run_enkf(X_obs, X0, n_ensemble=2)

    print("Running MFEnKF (2 HF & 1000 LF members)...")
    means_mfenkf, stds_mfenkf = run_mfenkf(X_obs, X0, hf_size=2, lf_size=1000)

    nodes_to_plot = [20, 50, 80] 
    obs_mapping = [2, 5, 8] 
    labels = ["Node at x=2.0", "Node at x=5.0", "Node at x=8.0"]

    def plot_time_series(title, means, stds, show_lf=False):
        plt.figure(figsize=(16, 9), dpi=200)
        for d in range(3):
            state_idx = nodes_to_plot[d]
            obs_idx = obs_mapping[d]
            
            plt.subplot(3, 1, d + 1)
            
            if d == 0: plt.title(title, fontsize=9)
            
            plt.plot(tspan, X_true[state_idx, :], color="black", lw=0.8, label="HF True State")
            if show_lf:
                plt.plot(tspan, X_lf_true[state_idx, :], color="blue", lw=0.8, linestyle='dashed', alpha=0.6, label="LF State")
            
            if means is not None:
                plt.plot(tspan, means[state_idx, :], color="tab:red", lw=1.2, label="Filter Mean")
                plt.fill_between(
                    tspan,
                    means[state_idx, :] - 2 * stds[state_idx, :],
                    means[state_idx, :] + 2 * stds[state_idx, :],
                    color="tab:red", alpha=0.2, label="±2σ"
                )
            
            plt.scatter(tspan, X_obs[obs_idx, :], color="tab:blue", s=2.0, label="Observed Data")
            
            plt.ylabel(labels[d], fontsize=7)
            
            plt.legend(loc="lower left", fontsize=5, framealpha=0.9) 
            
            if d == 2: plt.xlabel("Time (t)", fontsize=7)
            
        plt.tight_layout()

    print("Generating plots...")
    plot_time_series("1D Wave Target Dynamics: HF vs LF True States", None, None, show_lf=True)
    plot_time_series("1D Wave: EnKF Estimate vs. Ground Truth (2 ensemble members)", means_enkf, stds_enkf)
    plot_time_series("1D Wave: MFEnKF Estimate vs. Ground Truth (2 HF & 1000 LF ensemble members)", means_mfenkf, stds_mfenkf)
    
    plt.show()