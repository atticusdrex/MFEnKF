import numpy as np
import matplotlib.pyplot as plt

N = 100        
L = 10.0       
dx = L / N     
c = 1.0        
dt = 0.01      
t_end = 4.0
n_steps = int(t_end / dt)

x = np.linspace(0, L, N)
u0 = np.exp(-((x - L/2)**2))  
v0 = np.zeros(N)              
X = np.concatenate([u0, v0])  

def wave_1d_deriv_np(X, dx, c):
    N_pts = len(X) // 2
    u = X[:N_pts]
    v = X[N_pts:]
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / (dx**2)
    return np.concatenate([v, (c**2) * u_xx])

def rk4_step(X, dt, dx, c):
    k1 = wave_1d_deriv_np(X, dx, c)
    k2 = wave_1d_deriv_np(X + 0.5 * dt * k1, dx, c)
    k3 = wave_1d_deriv_np(X + 0.5 * dt * k2, dx, c)
    k4 = wave_1d_deriv_np(X + dt * k3, dx, c)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

print("Simulating wave propagation...")
snapshots = []
times_to_plot = [0, 100, 200, 300, 400] 

for i in range(n_steps + 1):
    if i in times_to_plot:
        snapshots.append((i * dt, X[:N])) 
    X = rk4_step(X, dt, dx, c)

plt.figure(figsize=(10, 6), dpi=150)
colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6']

for idx, (t, u_snap) in enumerate(snapshots):
    plt.plot(x, u_snap, label=f"t = {t:.1f}s", color=colors[idx], lw=2)

plt.title("1D Wave Equation: True State Forward Simulation")
plt.xlabel("Spatial Position (x)")
plt.ylabel("Displacement (u)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()