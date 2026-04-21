import sys
import jax
import jax.numpy as jnp
from jax import random as jrand

sys.path.append("..") 
from mfenkf.mfenkf import *

def wave_1d_deriv(X, dx, c=1.0):
    N = len(X) // 2
    u = X[:N]
    v = X[N:]
    
    u_xx = jnp.zeros_like(u)
    
    u_xx = u_xx.at[1:-1].set((u[:-2] - 2*u[1:-1] + u[2:]) / (dx**2))
    
    du_dt = v
    dv_dt = (c**2) * u_xx
    
    return jnp.concatenate([du_dt, dv_dt])