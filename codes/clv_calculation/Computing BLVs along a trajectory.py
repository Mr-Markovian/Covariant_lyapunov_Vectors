"""Computing BLVs along a trajectory"""
from jax import partial,jvp,vmap,jit,jacfwd
import jax.numpy as jnp
import numpy as np
import os
from numpy.linalg import norm,inv
from ode_solvers import rk4_solver as solver
#from scipy.integrate import solve_ivp 
from testbed_models import L96
qr_decomp=jnp.linalg.qr

# Compute the Jacobian of the system from the model
model_name='L96'
model_dim=10
num_clv=10

#L96 model parameters if needed( when you want to change the default ones)
forcing_=8.0

# these parameters now give you the mode, which is only a function of x, the state vector
model=partial(L96,forcing=forcing_)

#jacobian of the model
jac_model=jit(jacfwd(model))

# When we need to map as many vectors as the dimension of the model, this is efficient
@jit
def tangent_propagator(X_):
    """tangent propagator returns the rhs for the ode for the perturbations, does not change 
    the base point of the trajectory"""
    J_eval=jac_model(X_[:,0])
    Y_=J_eval@X_[:,1:]
    Z_=X_.at[:,1:].set(Y_)
    return Z_

# When the number of clvs is less than the jacobian dimension, we do not compute the full jacobian, but use jvp 
@jit
def model_plus_tangent_propagator(X_):
    "model(x) + jacobian(x)V for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(1), out_axes=(None,1))((in_tangents,)) 
    Y_=X_.at[:,0].set(y)
    Z_=Y_.at[:,1:].set(out_tangents)
    return Z_

# Reorthonormalization step of the output tangent vectors is done using this function
@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:])  # QR decomposition
    temp_=x_.at[:,1:].set(Q_)
    return temp_,Q_,R_

# Load the trajectory of the system
dt=0.01
dt_solver=0.002
n_iters=int(dt/dt_solver)

# Change to data path
start_idx=0 # starting point of the forward transient

# Solver for three regions
n1=20000  # number of time steps in forward transient

base_type='State'
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-10/State/')
base_traj=np.load('Multiple_trajectories_N=1_gap={}_ti=0.0_tf=500.0_dt_{}_dt_solver={}.npy'.format(dt,dt,dt_solver))[start_idx:,0]

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim)[:,:num_clv]  
X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
#Fixed jacobian in between observation gap
my_solver=jit(partial(solver,rhs_function=tangent_propagator,time_step=dt_solver))

#Store local growth rates, the G, R and C matrices
local_growth_rates=np.zeros((n1,num_clv))
Store_G=np.zeros((n1,model_dim,num_clv))
Store_R=np.zeros((n1,num_clv,num_clv))

for i in range(n1):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    X_start,Store_G[i],Store_R[i]=Output_tangent_QR(X_stop)
    X_start=X_start.at[:,0].set(base_traj[i+1])

np.save('BLV.npy',Store_G)
np.save('exponents.npy',local_growth_rates)
print('Job done')
