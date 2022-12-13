"""A vectorized code to compute multiple trajectories at simultaneously"""

from jax import partial,jvp,vmap,jit 
import jax.numpy as jnp
import numpy as np
from ode_solvers import rk4_solver as solver
from testbed_models import L96
import os
import json

forcing=8.0
# these parameters now give you the mode, which is only a function of x, the state vector
model=partial(L96,forcing=8.0)

model_dim=40
num_initials=1

# Generate intial conditions from a gaussian with variance =epsilon  
#epsilon=0.0001
#X_o=np.random.multivariate_normal(np.zeros(model_dim),epsilon*np.eye(model_dim),size=num_initials).T

# Or load a file containing initial conditions( like multiple ensembles)
with open('L96/L_96_I_x0_on_attractor_dim_40,forcing_8.json') as jsonfile:
    param=json.load(jsonfile)
#path to initials
X_o=param["initial"]

T_start=0.0
T_stop=500.0
dt=0.05
dt_solver=0.01
iters_dt=int(dt/dt_solver)
Trajs=np.zeros((int((T_stop-T_start)/dt),model_dim,num_initials))

@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    f_X= vmap(model,in_axes=(1),out_axes=(1))(X_) 
    return f_X    

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

@jit
def integrate_forward(X_):
    #forecast_ensemble=lax.fori_loop(0,iters_delta_t,my_solver,last_analysis_ensemble)
    X=X_
    for i in range(iters_dt):
        X=my_solver(x_initial=X)
    return X

if (num_initials==1):
    X_now=np.expand_dims(X_o,axis=1)
else:
    X_now=X_o


Trajs[0]=X_now
for i in range(1,Trajs.shape[0]):
    X_next=integrate_forward(X_now)
    Trajs[i]=X_next
    X_now=X_next

if (num_initials==1):
    Trajs=np.squeeze(Trajs,axis=-1)

os.chdir('L96')
#Save the solution
np.save('Multiple_trajectories_N={}_gap={}_ti={}_tf={}_dt_{}_dt_solver={}.npy'.format(num_initials,dt,T_start,T_stop,dt,dt_solver),Trajs)
print('job done')