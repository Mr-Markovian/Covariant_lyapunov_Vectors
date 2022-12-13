"""A vectorized code to compute multiple trajectories at simultaneously. The input has the shape
[time,dimension,no. of intial conditions]"""
from jax import partial,vmap,jit 
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ode_solvers import rk4_solver as solver
from testbed_models import L96
import os

forcing_=8.0
# these parameters now give you the mode, which is only a function of x, the state vector
model=partial(L96,forcing=forcing_)

seed_num=3
model_dim=10
num_initials=1

# Generate intial conditions from a gaussian with variance =epsilon  
#epsilon=0.00001
#X_o=np.random.multivariate_normal(np.zeros(model_dim),epsilon*np.eye(model_dim),size=num_initials).T
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/seed_{}'.format(model_dim,seed_num)
os.chdir(data_path)
X_o=np.expand_dims(np.load('Initial_condition_on_att.npy'),axis=1)

T_start=0.0
T_stop=500.0
dt=0.05
dt_solver=0.01
iters_delta_t=int(dt/dt_solver)

@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    f_X= vmap(model,in_axes=(1),out_axes=(1))(X_) 
    return f_X    

Trajs=np.zeros((int((T_stop-T_start)/dt),model_dim,num_initials))
my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
@jit
def integrate_forward(X_):
    #forecast_ensemble=lax.fori_loop(0,iters_delta_t,my_solver,last_analysis_ensemble)
    X=X_
    for i in range(iters_delta_t):
        X=my_solver(x_initial=X)
    return X

print(X_o.shape)
X_now=X_o
Trajs[0]=X_now
for i in range(1,Trajs.shape[0]):
    X_next=integrate_forward(X_now)
    Trajs[i]=X_next
    X_now=X_next

#Save the solution
print(Trajs[-1])
np.save('Multiple_trajectories_N={}_gap={}_ti={}_tf={}_dt_{}_dt_solver={}.npy'.format(num_initials,dt,T_start,T_stop,dt,dt_solver),Trajs[:,:,0])
print('job done')