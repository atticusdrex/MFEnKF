# %% Functions & imports 
import sys
sys.path.append("..")   # add parent folder (project/) to Python path
from mfenkf.mfenkf import * 

# operator term for lorenz system 
def lorenz_deriv(X, sigma=10, rho=28, beta=8/3):
    return jnp.array((
        sigma * (X[1] - X[0]), 
        X[0] * (rho - X[2]) - X[1], 
        X[0] * X[1] - beta * X[2]
    ))

# Simple function for discrete Euler steps 
def euler_step(X, dXdt, dt):
    return X + dXdt(X) * dt 

# euler step with random drift term 
def random_euler_step(key, X, dXdt, dt, var = 1.0):
    return X + dXdt(X) * dt + jnp.sqrt(var) * jrand.normal(key, shape=(3,))

# ── EnKF ─────────────────────────────────────────────────────────────────────
def enkf_forecast(keys, ensemble, dXdt, dt, process_var):
    """
    Propagate every ensemble member one step forward.

    Args:
        keys      : (N,) array of JAX PRNG keys, one per member
        ensemble  : (state_dim, N) array of ensemble members
        dXdt      : callable – deterministic RHS
        dt        : float – timestep
        process_var: float – isotropic process-noise variance Q = process_var * I

    Returns:
        (state_dim, N) forecasted ensemble
    """
    # vmap over columns: each member gets its own key
    def step_one(key, x):
        return random_euler_step(key, x, dXdt, dt, var=process_var)

    return jax.vmap(step_one, in_axes=(0, 1), out_axes=1)(keys, ensemble)


def enkf_analysis(key, ensemble_f, y, H, obs_var):
    """
    Perturbed-observation EnKF analysis step.

    Args:
        key        : JAX PRNG key for observation perturbations
        ensemble_f : (state_dim, N) forecasted ensemble
        y          : (obs_dim,)  observation vector
        H          : (obs_dim, state_dim) linear observation operator
        obs_var    : float – isotropic observation noise variance R = obs_var * I

    Returns:
        (state_dim, N) analysed ensemble
    """
    state_dim, N = ensemble_f.shape
    obs_dim      = y.shape[0]

    # ── ensemble statistics ───────────────────────────────────────────────
    x_mean = ensemble_f.mean(axis=1, keepdims=True)           # (state_dim, 1)
    A      = (ensemble_f - x_mean) / jnp.sqrt(N - 1)         # anomaly matrix

    P_xy = A @ (H @ A).T                                      # (state_dim, obs_dim)  = P H^T  (scaled)
    P_yy = (H @ A) @ (H @ A).T + obs_var * jnp.eye(obs_dim)  # (obs_dim,  obs_dim )  = H P H^T + R

    # ── Kalman gain: K = P H^T (H P H^T + R)^{-1} ───────────────────────
    K = jnp.linalg.solve(P_yy.T, P_xy.T).T                   # (state_dim, obs_dim)

    # ── perturbed observations ε ~ N(0, R) ────────────────────────────────
    eps = jnp.sqrt(obs_var) * jrand.normal(key, shape=(obs_dim, N))

    # ── analysis update ──────────────────────────────────────────────────
    innovation = (y[:, None] + eps) - H @ ensemble_f          # (obs_dim, N)
    return ensemble_f + K @ innovation                        # (state_dim, N)


def run_enkf(X_obs, X0_true, dt, process_var, obs_var,
             n_ensemble=50, H=None, seed=0):
    """
    Run EnKF over the full observation sequence.

    Args:
        X_obs      : (state_dim, n_steps) noisy observations
        X0_true    : (state_dim,)         true initial state
        dt         : float
        process_var: float  – process noise variance
        obs_var    : float  – observation noise variance
        n_ensemble : int    – number of ensemble members
        H          : observation matrix; defaults to full identity (observe all states)
        seed       : int    – PRNG seed

    Returns:
        means : (state_dim, n_steps)  posterior mean at each step
        stds  : (state_dim, n_steps)  posterior std  at each step
    """
    state_dim, n_steps = X_obs.shape
    if H is None:
        H = jnp.eye(state_dim)                                # observe everything
    obs_dim = H.shape[0]

    # ── initialise ensemble around the first observation ─────────────────
    key = jrand.PRNGKey(seed)
    key, subkey = jrand.split(key)
    ensemble = X_obs[:, 0:1] + jnp.sqrt(obs_var) * jrand.normal(
        subkey, shape=(state_dim, n_ensemble))                 # (state_dim, N)

    means = np.zeros((state_dim, n_steps))
    stds  = np.zeros((state_dim, n_steps))
    means[:, 0] = ensemble.mean(axis=1)
    stds[:,  0] = ensemble.std(axis=1)

    for i in tqdm(range(1, n_steps)):
        # split keys: one per ensemble member for forecast + one for analysis
        key, fkey, akey = jrand.split(key, 3)
        fkeys = jrand.split(fkey, n_ensemble)               # (N, 2)

        # forecast
        ensemble = enkf_forecast(fkeys, ensemble, lorenz_deriv, dt, process_var)

        # analysis
        y = X_obs[:obs_dim, i]                               # current observation
        ensemble = enkf_analysis(akey, ensemble, y, H, obs_var)

        means[:, i] = ensemble.mean(axis=1)
        stds[:,  i] = ensemble.std(axis=1)

    return jnp.array(means), jnp.array(stds)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── simulation parameters (unchanged from your code) ─────────────────
    X0         = jnp.ones(3)
    t_end      = 2.0
    dt         = 1e-2
    n_steps    = int(jnp.ceil(t_end / dt))
    tspan      = jnp.linspace(0, t_end, n_steps)
    process_var = 3e-1
    observ_var  = 20.0

    # %% ── generate ground truth and noisy observations ──────────────────────
    X = np.zeros((3, n_steps));  X[:, 0] = X0
    for i in range(1, n_steps):
        X[:, i] = euler_step(X[:, i-1], lorenz_deriv, dt)

    Xr = np.zeros((3, n_steps));  Xr[:, 0] = X0; 
    Xobs = np.zeros_like(Xr); Xobs[:,0] = X0; 
    proc_keys = jrand.split(jrand.PRNGKey(42), n_steps)
    obsv_keys = jrand.split(jrand.PRNGKey(43), n_steps)
    for i in range(1, n_steps):
        Xr[:, i]   =  random_euler_step(proc_keys[i], Xr[:, i-1], lorenz_deriv, dt, var=process_var)
        Xobs[:, i] = Xr[:,i] + jrand.normal(obsv_keys[i], shape=(3,)) * jnp.sqrt(observ_var)

    # %% ── run EnKF ──────────────────────────────────────────────────────────
    print("Running EnKF …")
    means, stds = run_enkf(
        jnp.array(Xobs), X0,
        dt          = dt,
        process_var = process_var,
        obs_var     = observ_var,
        n_ensemble  = 5,
    )
    print("Done.")

    # %% ── plot results ──────────────────────────────────────────────────────
    labels = ["x(t)", "y(t)", "z(t)"]
    figure(figsize=(12, 12), dpi=200)
    for d in range(3):
        subplot(3, 1, d + 1)
        if d == 0:
            title("EnKF estimate vs ground truth")
        # plot(tspan, X[d, :],       color="tab:blue",   lw=0.8, label="truth")
        plot(tspan, Xr[d, :],      color="black", lw=0.5, alpha=1.0, linestyle = 'solid', label="True State")
        plot(tspan, means[d, :],   color="tab:red",    lw=1.0, label="EnKF mean")
        fill_between(
            tspan,
            means[d, :] - 2 * stds[d, :],
            means[d, :] + 2 * stds[d, :],
            color="tab:red", alpha=0.2, label="±2σ"
        )
        scatter(tspan, Xobs[d,:], color = "tab:blue", s = 1.0, alpha = 1.0, label = "Observed Data")
        ylabel(labels[d]);  legend(loc="upper right", fontsize=7)
        if d == 2:
            xlabel("Time (t)")

        
        


