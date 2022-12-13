"""Computing CLVs along a trajectory by Ginelli's Algorithm.
We do not evolve the system, but only use the points on the trajectory to evolve
the perturbations in the tangent space."""

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
model_dim=40
num_clv=20

#L63 model parameters if needed( when you want to change the default ones)
#rho = 28.0
#sigma = 10.0
#beta = 8.0/3

#L96 model parameters if needed( when you want to change the default ones)
forcing=8.0

# these parameters now give you the mode, which is only a function of x, the state vector
model=partial(L96,forcing)

#jacobian of the model
jac_model=jit(jacfwd(model))

# When we need to map as many vectors as the dimension of the model, this is efficient
@jit
def tangent_propagator(X_):
    """tangent propagator returns the rhs for the ode for the perturbations, does not change 
    the base point of the trajectory"""
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(0), out_axes=(None,0))((in_tangents,))
    Y_=X_.at[:,1:].set(out_tangents) 
    return Y_

# When the number of clvs is less than the jacobian dimension, we do not compute the full jacobian, but use jvp 
@jit
def model_plus_tangent_propagator(X_):
    "model(x) + jacobian(x)V for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(0), out_axes=(None,0))((in_tangents,)) 
    Y_=X_.at[:,0].set(y)
    Y_=X_.at[:,1:].set(out_tangents)
    return Y_

# Reorthonormalization step of the output tangent vectors is done using this function
@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:])  # QR decomposition
    temp_=x_.at[:,1:].set(Q_)
    return temp_,Q_,R_

# The backward evolution computing the clv coefficients 
def backward_evolution(x_,y_k):
    """Generates coeffients at the previous time step from the current time step"""
    temp_C=inv(x_)@y_k
    #row_sums = temp_.sum(axis=1,keepdims=True)
    row_sums=np.linalg.norm(temp_C,ord=2,axis=0,keepdims=True)
    temp_C = temp_C / row_sums
    return temp_C

# Forward transient

# Load the trajectory of the system
dt=0.05
dt_solver=0.01
n_iters=int(dt/dt_solver)

# Change to data path
start_idx=25000 # starting point of the forward transient

base_type='State'
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
base_traj=np.load('State_g={}.npy'.format(dt))[start_idx:]

base_type='state_noisy'
mu=81.0
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/mu={}'.format(mu))
base_traj=np.load('{}_g={}_mu={}.npy'.format(base_type,dt,mu))

#base_type='Analysis'
#mu=1.0
#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/Analysis')
#base_traj=np.load('Analysis_mean_g={}_mu={}.npy'.format(dt,mu))[start_idx:]

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim) #   [:,:num_clv]  
X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
#Fixed jacobian in between observation gap
my_solver=jit(partial(solver,rhs_function=tangent_propagator,time_step=dt_solver))

#Evolving jacobian in between observation gap, next thing to see for higher observation gaps.
#my_solver=jit(partial(solver,rhs_function=model_plus_tangent_propagator,time_step=dt_solver))

# Solver for three regions
n1=10000  # number of time steps in forward transient
n2=5000  # number of time steps in the interval
n3=10000  # number of time steps in backward transient

#Store local growth rates, the G, R and C matrices
local_growth_rates=np.zeros((n2+n3,num_clv))
Store_G=np.zeros((n2,model_dim,num_clv))
Store_R=np.zeros((n2+n3,num_clv,num_clv))
Store_C=np.zeros((n2,num_clv,num_clv))

for i in range(n1):
    for j in range(n_iters):
       X_stop=my_solver(x_initial=X_start)
       X_start=X_stop
    #Compute the exponents and store the matrices
    #local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=1))/dt
    X_start,_,_=Output_tangent_QR(X_stop)
    X_start=X_start.at[:,0].set(base_traj[i+1])

print('Transient is over, moving on to the interval in question')
# Interval where we sample the clvs.

for i in range(n2):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    X_start,Store_G[i],Store_R[i]=Output_tangent_QR(X_stop)
    X_start=X_start.at[:,0].set(base_traj[i+1])

print('The matrics of backward lyapunov vectors stored, moving on to backward transient part')
# #Free Up space
np.save('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_G)  
del Store_G  

# Backward transent interval begins, where going forward over the backward transient interval
# Store Q and R matrices
for i in range(n3):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    local_growth_rates[n2+i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    X_start,_,Store_R[n2+i]=Output_tangent_QR(X_stop)
    X_start=X_start.at[:,0].set(base_traj[i+1])

#  Free up space
np.save('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),local_growth_rates)
del local_growth_rates

print('Moving backward on the backward transient part, to converge to the clvs')   

# A generic upper triangular matirx for the backward transient
C_tilde_initial=np.zeros((num_clv,num_clv))
for i in range(num_clv):
    C_tilde_initial[i,i:]=np.ones(num_clv-i)
C_tilde=C_tilde_initial
print(C_tilde)

#  Going backwards over the backward transient via backward evolution using stored R-inverse
for i in range(n3):
    l=n2+n3-1-i
    C_tilde=backward_evolution(Store_R[l],C_tilde)

# CLV on the interval of interest 
print('Sampling the clvs over the interval of interest')

# # Sampling the clvs(i.e. in terms of coefficients when expressed in BLV) 
for i in range(n2):
    l=n2-1-i
    C_tilde=backward_evolution(Store_R[l],C_tilde)
    Store_C[l]=C_tilde

print('Sampled, now storing all data')
np.save('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_C)
print('Job done')
