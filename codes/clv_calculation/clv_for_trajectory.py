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
from testbed_models import L63
qr_decomp=jnp.linalg.qr

# Compute the Jacobian of the system from the model
model=L63       # use the model imported above
model_dim=3
num_clv=3

#model parameters if needed
rho = 28.0
sigma = 10.0
beta = 8.0/3

# these parameters now give you the mode, which is only a function of x, the state vector
model=partial(L63,sigma=10.0,rho=28.0,beta=8./3)

#jacobian of the model
jac_model=jit(jacfwd(model))

# When we need to map as many vectors as the dimension of the model, this is efficient
@jit
def tangent_propagator(X_):
    """tangent propagator returns the rhs for the ode for the perturbations, does not change 
    the base point of the trajectory"""
    J_eval=jac_model(X_[:,0])
    Y_=X_.at[:,1:].set(J_eval@(X_[:,1:]))
    return Y_

# When the number of clvs is less than the jacobian dimension, we do not compute the full jacobian, but use jvp 


# Reorthonormalization step of the output tangent vectors is done using this function
@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:])  # QR decomposition
    #cn=np.linalg.cond(jnp.transpose(temp_[1:,:])) 
    #local_growth_rates_= np.sum(np.absolute(R_),axis=1), should give same ans
    temp_=x_.at[:,1:].set(Q_)
    return temp_,Q_,R_

# The backward evolution computing the clv coefficients 
def backward_evolution(x_,y_k):
    """Generates coeffients at the previous time step from the current time step"""
    temp_=inv(x_)@y_k
    #row_sums = temp_.sum(axis=1,keepdims=True)
    row_sums=np.linalg.norm(temp_,ord=2,axis=0,keepdims=True)
    temp_ = temp_ / row_sums
    return temp_

# Forward transient

# Load the trajectory of the system
dt=0.01     # and 0.05
dt_solver=0.002
n_iters=int(dt/dt_solver)

# Change to data path
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
base_traj=np.load('Trajectory_{}_T=500.npy'.format(dt))

#Initial condition of the transient trajectory
start_idx=20000, # starting point of the forward transient

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim)[:num_clv]  
X_start[:,0]=base_traj[0]

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
my_solver=partial(solver,rhs_function=tangent_propagator,time_step=dt_solver)

n1=10000
n2=10000
n3=10000

#Store local growth rates, the G, R and C matrices
local_growth_rates=np.zeros((n1+n2+n3-1,num_clv))
#Store_G=np.zeros((n2-1,model_dim,num_clv))
#Store_R=np.zeros((n2+n3-2,num_clv,num_clv))
#Store_C=np.zeros((n2-1,num_clv,num_clv))

for i in range(n1-1):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    local_growth_rates[n1-1+j]=np.log(norm(X_stop[:,1:],axis=1))/dt
    X_start,_,_=Output_tangent_QR(X_stop)
    X_start=X_start.at[:,0].set(base_traj[i+1])


print('Transient is over, moving on to the interval in question')
# Interval where we sample the clvs.

# for j in range(n2-1):
#     X_stop=jit(my_solver(X_start,dt))
#     #Compute the exponents and store the matrices
#     local_growth_rates[n1-1+j]=np.log(norm(X_stop[1:],axis=1))/dt
#     X_start,Store_G[j],Store_R[j]=Output_tangent_QR(X_stop)

# print('The matrics of backward lyapunov vectors stored, moving on to backward transient part')

# #Free Up space
# np.save('matrices_g_{}_seed_{}.npy'.format(model_dim,seed_num),Store_G)  
# del Store_G  

# # Backward transent interval begins, where going forward over the backward transient interval
# # Store Q and R matrices
# for j in range(n3-1):
#     X_stop=jit(my_solver(X_start,dt))
#     #Compute the exponents and store the matrices
#     local_growth_rates[n1+n2-2+j]=np.log(norm(X_stop[1:],axis=1))/dt
#     X_start,_,Store_R[n2-1+j]=Output_tangent_QR(X_stop)

# # Free up space
# np.save('local_growth_rates_{}.npy'.format(model_dim),local_growth_rates)
# del local_growth_rates

# print('Moving backward on the backward transient part, to converge to the clvs')   

# # A generic upper triangular matirx for the backward transient
# C_tilde_initial=np.zeros((num_clv,num_clv))
# for i in range(num_clv):
#     C_tilde_initial[i,i:]=np.ones(num_clv-i)
# C_tilde=C_tilde_initial

# # Going backwards over the backward transient via backward evolution using stored R-inverse
# for j in range(n3-1):
#     l=n2+n3-3-j
#     C_tilde=backward_evolution(Store_R[l],C_tilde)

# # CLV on the interval of interest 
# print('Sampling the clvs over the interval of interest')

# # Sampling the clvs(i.e. in terms of coefficients when expressed in BLV) 
# for j in range(n2-1):
#     l=n2-2-j
#     C_tilde=backward_evolution(Store_R[l],C_tilde)
#     Store_C[l]=C_tilde

# print('Sampled, now storing all data')
# np.save('matrices_c_{}_seed_{}.npy'.format(model_dim,seed_num),Store_C)
print('Job done')
