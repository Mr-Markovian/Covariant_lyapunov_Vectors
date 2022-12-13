"""Computing CLVs along a trajectory by Ginelli's Algorithm.
We do not evolve the system, but only use the points on the trajectory to evolve
the perturbations in the tangent space."""

from jax import partial,jvp,vmap,jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import os
from numpy.linalg import norm,inv
from ode_solvers import rk4_solver as solver
from testbed_models import L96
qr_decomp=jnp.linalg.qr

#L96 model parameters if needed( when you want to change the default ones)
forcing_=8.0

# these parameters now give you the mode, which is only a function of x, the state vector
model=jit(partial(L96,forcing=forcing_))

# When we need to map as many vectors as the dimension of the model, this is efficient
@jit
def tangent_propagator(X_):
    """tangent propagator returns the rhs for the ode for the perturbations, does not change 
    the base point of the trajectory"""
    x_eval=X_[:,0]  # Nx1
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp,model , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(1), out_axes=(None,1))((in_tangents,)) 
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
def backward_evolution(R_,C_next):
    """Generates coeffients at the previous time step from the current time step"""
    temp_C=inv(R_)@C_next
    #row_sums = temp_.sum(axis=1,keepdims=True)
    row_sums=np.linalg.norm(temp_C,ord=2,axis=0,keepdims=True)
    temp_C = temp_C / row_sums
    return temp_C

model_name='L96'
model_dim=40
num_clv=model_dim

# Forward transient
seed_num=3

# Load the trajectory of the system
dt=0.05
dt_solver=0.01
n_iters=int(dt/dt_solver)

# Change to data path
start_idx=0 # starting point of the forward transient

# Solver for three regions
n1=4000           # forward transient
n2=4000           # steps in the interval
n3=4000           # backward transient

# Change to data path
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/explanation'.format(model_dim)
os.chdir(data_path)

base_type='state_noisy'
#sigma=0.1

# base_type='Analysis'
# mu=1.0
# os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/Analysis')
# base_traj=np.load('Analysis_mean_g={}_mu={}.npy'.format(dt,mu))[start_idx:]


# orthogonal perturbations at t=0
X_start1=jnp.zeros((model_dim,num_clv+1))
X_start=X_start1.at[:,1:].set(np.eye(model_dim)[:,:num_clv])  

# A generic upper triangular matirx for the backward transient which is needed always
C_tilde_initial=np.zeros((num_clv,num_clv))
for i in range(num_clv):
    C_tilde_initial[i,i:]=np.ones(num_clv-i)
C_tilde=C_tilde_initial
print(C_tilde)

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
my_solver=jit(partial(solver,rhs_function=tangent_propagator,time_step=dt_solver))

#Store local growth rates, the G, R and C matrices
base_type='state_noisy'

sigmas=np.array([0.0,0.1,0.2,0.3,0.4,0.5])
for sigma in sigmas:
    local_growth_rates=np.zeros((n2+n3,num_clv))
    Store_G=np.zeros((n2,model_dim,num_clv))
    Store_R=np.zeros((n2+n3,num_clv,num_clv))
    Store_C=np.zeros((n2,num_clv,num_clv))
    #os.chdir(data_path+'/sigma={}'.format(sigma))
    os.chdir(data_path+'/analysis_sigma={}'.format(sigma))
    base_traj=np.load('{}_g={}_sigma={}.npy'.format(base_type,dt,sigma))

    # forward transient
    for i in range(n1):
        X_start=X_start.at[:,0].set(base_traj[i])
        for j in range(n_iters):
            X_stop=my_solver(x_initial=X_start)
            X_start=X_stop
        X_start,_,_=Output_tangent_QR(X_stop)

    # the interval over which we want to calculate CLVs
    for i in range(n2):
        X_start=X_start.at[:,0].set(base_traj[n1+i])
        for j in range(n_iters):
            X_stop=my_solver(x_initial=X_start)
            X_start=X_stop
        #Compute the exponents and store the matrices
        local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
        X_start,Store_G[i],Store_R[i]=Output_tangent_QR(X_stop)

    # Free Up space
    np.save('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_G)  
    del Store_G  

    # Backward transent interval begins
    # Store R matrices
    for i in range(n3):
        X_start=X_start.at[:,0].set(base_traj[n1+n2+i])
        for j in range(n_iters):
            X_stop=my_solver(x_initial=X_start)
            X_start=X_stop
        #Compute the exponents and store the matrices
        local_growth_rates[n2+i]=np.log(norm(X_stop[:,1:],axis=0))/dt
        X_start,_,Store_R[n2+i]=Output_tangent_QR(X_stop)

    #  Free up space
    np.save('local_growth_rates_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),local_growth_rates)
    del local_growth_rates

    #  Going backwards over the backward transient via backward evolution using stored R-inverse
    for i in range(n3):
        l=n2+n3-1-i
        C_tilde=backward_evolution(Store_R[l],C_tilde)

    # # Sampling the clvs(i.e. in terms of coefficients when expressed in BLV) 
    for i in range(n2):
        l=n2-1-i
        C_tilde=backward_evolution(Store_R[l],C_tilde)
        Store_C[l]=C_tilde

    np.save('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type),Store_C)
    print('Job done')
