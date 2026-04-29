import numpy as np
import matplotlib.pyplot as plt
import os

print("Initializing 1D Wave Equation Time-Series Simulation...")

N_pts_hf = 100
N_pts_lf = 50
state_dim_hf = 2 * N_pts_hf
state_dim_lf = 2 * N_pts_lf
L = 10.0
dx_hf = L / N_pts_hf
dx_lf = L / N_pts_lf

t_end = 5.0
dt = 0.01
n_steps = int(np.ceil(t_end / dt))
tspan = np.linspace(0, t_end, n_steps)
process_var = 1e-4
obs_var = 1e-2

x_hf = np.linspace(0, L, N_pts_hf)
x_lf = np.linspace(0, L, N_pts_lf)

obs_indices_hf = np.arange(0, N_pts_hf, 10)
obs_indices_lf = np.arange(0, N_pts_lf, 5) 
obs_dim = len(obs_indices_hf)

H_hf = np.zeros((obs_dim, state_dim_hf))
for i, idx in enumerate(obs_indices_hf): H_hf[i, idx] = 1.0

H_lf = np.zeros((obs_dim, state_dim_lf))
for i, idx in enumerate(obs_indices_lf): H_lf[i, idx] = 1.0

def wave_deriv(X, dx, c):
    N_pts = len(X) // 2
    u, v = X[:N_pts], X[N_pts:]
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / (dx**2)
    return np.concatenate([v, (c**2) * u_xx])

def hf_deriv(X): return wave_deriv(X, dx_hf, c=1.0)
def lf_deriv(X): return wave_deriv(X, dx_lf, c=1.0)

def prolong(X_lf):
    u_lf, v_lf = X_lf[:N_pts_lf], X_lf[N_pts_lf:]
    u_hf = np.interp(x_hf, x_lf, u_lf)
    v_hf = np.interp(x_hf, x_lf, v_lf)
    return np.concatenate([u_hf, v_hf])

def restrict(X_hf):
    u_hf, v_hf = X_hf[:N_pts_hf], X_hf[N_pts_hf:]
    u_lf = np.interp(x_lf, x_hf, u_hf)
    v_lf = np.interp(x_lf, x_hf, v_hf)
    return np.concatenate([u_lf, v_lf])

def prolong_ensemble(ens_lf):
    return np.apply_along_axis(prolong, 0, ens_lf)

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
    ens = X0_true[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim_hf, n_ensemble))
    means, stds = np.zeros((state_dim_hf, n_steps)), np.zeros((state_dim_hf, n_steps))
    means[:, 0], stds[:, 0] = ens.mean(axis=1), ens.std(axis=1)

    for i in range(1, n_steps):
        ens = enkf_forecast(ens, hf_deriv, dt, process_var)
        ens = enkf_analysis(ens, X_obs[:, i], H_hf, obs_var)
        means[:, i], stds[:, i] = ens.mean(axis=1), ens.std(axis=1)
    return means, stds

def run_mfenkf(X_obs, X0_true_hf, hf_size, lf_size):
    X0_true_lf = restrict(X0_true_hf)
    
    hf_ens = X0_true_hf[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim_hf, hf_size))
    
    linked_ens = np.zeros((state_dim_lf, hf_size))
    for j in range(hf_size):
        linked_ens[:, j] = restrict(hf_ens[:, j])
        
    lf_ens = X0_true_lf[:, None] + np.sqrt(obs_var) * np.random.normal(size=(state_dim_lf, lf_size))

    means, stds = np.zeros((state_dim_hf, n_steps)), np.zeros((state_dim_hf, n_steps))
    means[:, 0] = hf_ens.mean(axis=1)
    stds[:, 0] = prolong_ensemble(lf_ens).std(axis=1) + (hf_ens - prolong_ensemble(linked_ens)).std(axis=1)

    for i in range(1, n_steps):
        # Forecasts
        hf_ens = enkf_forecast(hf_ens, hf_deriv, dt, process_var)
        linked_ens = enkf_forecast(linked_ens, lf_deriv, dt, process_var)
        lf_ens = enkf_forecast(lf_ens, lf_deriv, dt, process_var)
        
        # Analyses
        y = X_obs[:, i]
        hf_ens = enkf_analysis(hf_ens, y, H_hf, obs_var)
        linked_ens = enkf_analysis(linked_ens, y, H_lf, obs_var)
        lf_ens = enkf_analysis(lf_ens, y, H_lf, obs_var)
        
        # Prolong LF means up to HF dimensions for math
        hf_mean = hf_ens.mean(axis=1)
        lf_mean_prolonged = prolong(lf_ens.mean(axis=1))
        linked_mean_prolonged = prolong(linked_ens.mean(axis=1))

        # Calculate MF mean and standard deviation
        means[:, i] = hf_mean + (lf_mean_prolonged - linked_mean_prolonged)
        stds[:, i] = prolong_ensemble(lf_ens).std(axis=1) + (hf_ens - prolong_ensemble(linked_ens)).std(axis=1)
        
    return means, stds

if __name__ == "__main__":
    print("Generating ground truth and sparse observations...")
    X0_hf = np.concatenate([np.exp(-((x_hf - L/2)**2)), np.zeros(N_pts_hf)])
    
    X_true = np.zeros((state_dim_hf, n_steps))
    X_lf_true = np.zeros((state_dim_hf, n_steps))
    X_obs = np.zeros((obs_dim, n_steps))
    
    X_true[:, 0] = X0_hf
    X_lf_true[:, 0] = X0_hf
    X_obs[:, 0] = H_hf @ X0_hf

    curr_X_hf = X0_hf
    curr_X_lf = restrict(X0_hf)
    
    for i in range(1, n_steps):
        curr_X_hf = random_rk4_step(curr_X_hf, hf_deriv, dt, var=process_var)
        curr_X_lf = random_rk4_step(curr_X_lf, lf_deriv, dt, var=process_var)
        
        X_true[:, i] = curr_X_hf
        X_lf_true[:, i] = prolong(curr_X_lf)
        X_obs[:, i] = H_hf @ curr_X_hf + np.sqrt(obs_var) * np.random.normal(size=(obs_dim,))

    print("Running Standard EnKF (2 members)...")
    means_enkf, stds_enkf = run_enkf(X_obs, X0_hf, n_ensemble=2)

    print("Running MFEnKF (2 HF & 1000 LF members)...")
    means_mfenkf, stds_mfenkf = run_mfenkf(X_obs, X0_hf, hf_size=2, lf_size=1000)

    nodes_to_plot = [20, 50, 80] 
    obs_mapping = [2, 5, 8] 
    labels = ["Node at x=2.0", "Node at x=5.0", "Node at x=8.0"]

    def plot_time_series(title, means, stds, show_lf=False):
        import os
        
        fig = plt.figure(figsize=(16, 9), dpi=200)
        
        fig.suptitle(title, fontsize=16)

        for d in range(3):
            state_idx = nodes_to_plot[d]
            obs_idx = obs_mapping[d]
            
            ax = plt.subplot(3, 1, d + 1)
            
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
            
            plt.ylabel(labels[d], fontsize=12, labelpad=12)
            plt.legend(loc="lower left", fontsize=9, framealpha=0.9) 
            
            plt.tick_params(axis='both', which='major', labelsize=10)
            
            plt.xlabel("Time (s)", fontsize=12, labelpad=8)
                
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        desktop_path = os.path.expanduser("~/Desktop")
        safe_filename = title.replace(":", "").replace(" ", "_").replace(".", "")
        full_save_path = os.path.join(desktop_path, f"{safe_filename}.png")
        
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {full_save_path}")

    print("Generating plots...")
    plot_time_series("1D Wave Target Dynamics: HF vs LF True States", None, None, show_lf=True)
    plot_time_series("1D Wave: EnKF Estimate vs. Ground Truth (2 ensemble members)", means_enkf, stds_enkf)
    plot_time_series("1D Wave: MFEnKF Estimate vs. Ground Truth (2 HF & 1000 LF ensemble members)", means_mfenkf, stds_mfenkf)
    
    plt.show()