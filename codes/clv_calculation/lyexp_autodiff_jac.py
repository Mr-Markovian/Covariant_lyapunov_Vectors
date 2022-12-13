"""This script implements dynamic algorithm for computation of covariant lyapunov
vector of a dynamical system. We evolve first for a transient period evolution
using a set of perturbation vectors along a trajectory so that they converge to forward 
lyapunov vectors. Then, we perform evolution with storing the gram-schmidt vectors
Q and the upper-triangular matrix R at all the time steps at which we want to compute
the vectors later. """

"""First you need to make sure that JAX is installed by
'pip install jaxlib', followed by, 'pip install -e'"""

# Lyapunov Exponents for a system of ODE
from jax import partial,jvp,vmap,jit 
import jax.numpy as jnp
import numpy as np
import os
from numpy.linalg import norm
from ode_solvers import rk4_solver as solver
import json 
import matplotlib.pyplot as plt
from testbed_models import L96

qr_decomp=jnp.linalg.qr
diagonal=np.diag

#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63_clvs/noisy_state/State')
#Model Details.....................................................................

model=L96
model_dim=40
forcing=8
num_lexp=40
print('To calculate first {} covariant lyapunov vectors '.format(num_lexp))

#Model details for L63
#sigma,rho,beta=10.,28.,8./3

# time step for reorthonormalsation
dt=0.02
dt_solver=0.01
n_iters=int(dt/dt_solver)
T_i=0
T_f=1000
total_steps=int((T_f-T_i)/dt)          

# T2 is going to be used twice, first during forward run, second during sampling the clvs 

""" Choice of initial vectors here is standard orthonormal vectors """
with open('L96/L_96_I_x0_on_attractor_dim_40,forcing_8.json') as jsonfile:
    param=json.load(jsonfile)

#Initial condition of the transient trajectory
X_start=np.zeros((model_dim,num_lexp+1))

X_start[:,0]=param["initial"]
X_start[:,1:]=np.eye(model_dim)[:,:num_lexp]  # orthogonal perturbations

#Store local growth rates
local_growth_rates=np.zeros((total_steps,num_lexp))

@jit
def model_plus_jacobian(X_):
    "model + jacobian for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  
    in_tangents=X_[:,1:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(1), out_axes=(None,1))((in_tangents,)) 
    Y_=X_.at[:,0].set(y)
    Z_=Y_.at[:,1:].set(out_tangents)
    return Z_

@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(x_[:,1:]) 
    temp_=x_.at[:,1:].set(Q_)
    return temp_,Q_,R_

my_solver=jit(partial(solver,rhs_function=model_plus_jacobian,time_step=dt_solver))

# For the initial transient, we now integrate this,Storing the trajectory and the exponents:
for i in range(total_steps):
    for j in range(n_iters):
       X_stop=my_solver(x_initial=X_start)
       X_start=X_stop
    #Compute the exponents and store the matrices
    local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    X_start,_,_=Output_tangent_QR(X_stop)

lexp=np.mean(local_growth_rates,axis=0)
np.save('lexp_T={}_dt={}.npy'.format(T_f-T_i,dt),lexp)

plt.figure(figsize=(16,8))
plt.plot(np.arange(1,num_lexp+1),lexp)
plt.xlabel(r'$i$')
plt.show()

# For the interval we want to sample over
print('Job done')
