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
def euler_step(key, X, dXdt, dt):
    return X + dXdt(X) * dt 

# euler step with random drift term 
def random_euler_step(key, X, dXdt, dt, var = 1.0):
    return X + dXdt(X) * dt + var * jrand.normal(key, shape=(3,))

if __name__ == "__main__":
    X0 = jnp.ones(3) # Initial conditions of all ones 
    t_end = 20.0 # end time 
    dt = 1e-3 # time discretization 
    n_steps = int(jnp.ceil(t_end / dt)) # number of steps 
    tspan = jnp.linspace(0,t_end, n_steps) # making a time vector
    process_var = 3e-1 # variance of the process 
    observ_var = 1e-2 # observation variance 
    # %% Testing simulation 
    X = np.zeros((3, n_steps))
    X[:,0] = X0 # setting initial conditions 
    for i in range(1, n_steps):
        X[:,i] = euler_step(X[:,i-1], lorenz_deriv, dt)

    figure(figsize=(10,5), dpi = 200)
    subplot(3,1,1) 
    title("Lorenz System Trajectory")
    plot(tspan, X[0,:])
    ylabel("x(t)")
    subplot(3,1,2) 
    plot(tspan, X[1,:])
    ylabel("y(t)")
    subplot(3,1,3) 
    plot(tspan, X[2,:])
    xlabel("Time (t) ")
    ylabel("z(t)")

    # %% testing random simulation 
    Xr = np.zeros((3, n_steps))
    Xr[:,0] = X0 # setting initial conditions 
    keys = jrand.split(jrand.PRNGKey(42), n_steps)
    for i in range(1, n_steps):
        Xr[:,i] = random_euler_step(keys[i], Xr[:,i-1], lorenz_deriv, dt, var = 3e-1)

    figure(figsize=(10,6), dpi = 200)
    subplot(3,1,1) 
    title("Lorenz System Trajectory")
    plot(tspan, X[0,:], label = "unperturbed")
    plot(tspan, Xr[0,:], label = "perturbed")
    ylim(-20, 40)
    legend()
    ylabel("x(t)")
    subplot(3,1,2) 
    plot(tspan, X[1,:])
    plot(tspan, Xr[1,:])
    ylabel("y(t)")
    subplot(3,1,3) 
    plot(tspan, X[2,:])
    plot(tspan, Xr[2,:])
    xlabel("Time (t) ")
    ylabel("z(t)")


