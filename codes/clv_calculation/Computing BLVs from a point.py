"""This script evolve a set of perturbation vectors along a trajectory so that they converge to backward 
lyapunov vectors. First you need to make sure that JAX is installed by
'pip install jaxlib', followed by, 'pip install -e'"""

# Lyapunov Exponents for a system of ODE
from jax import partial,jvp,vmap,jit,jacfwd 
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import os
from ode_solvers import rk4_solver as solver
from numpy.linalg import norm
from testbed_models import L96

qr_decomp=jnp.linalg.qr
diagonal=np.diag
seed_num=2111997
np.random.seed(seed_num)

os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-10/BLV_convergence')

#Model Details.....................................................................
#L96 model parameters if needed( when you want to change the default ones)
forcing_=8.0
model=L96
model_dim=10
num_clv=5
model=partial(L96,forcing=forcing_)
print('To calculate first {} covariant lyapunov vectors '.format(num_clv))

#jacobian of the model
jac_model=jit(jacfwd(model))

# Trajectory details 
dt=0.02
dt_solver=0.004     
n_iters=int(dt/dt_solver)

# number of time-steps to integrate forward
n1=10000

#Store local growth rates ,Store G and R matrices
local_growth_rates=np.zeros((n1,num_clv))
Store_G=np.zeros((n1,model_dim,num_clv))

def fun(t,x):
    "RHS for scipy.integrate.solve_ivp"
    return np.asarray(model_plus_jacobian(x))

@jit
def model_plus_jacobian(X_):
    "model + jacobian for state space and tanegent linear evolution at a point"
    x_eval=X_[:,0]  # Nx1
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
    #cn=np.linalg.cond(jnp.transpose(temp_[1:,:])) 
    #local_growth_rates_= np.sum(np.absolute(R_),axis=1), should give same ans
    temp_=x_.at[:,1:].set(Q_)
    return temp_,Q_,R_

#Initial condition of the transient trajectory
X_o=np.load('Initial_condition_on_att.npy')

# orthogonal perturbations at t=0
X_start=np.zeros((model_dim,num_clv+1))
X_start[:,1:]=np.eye(model_dim)[:,:num_clv]  
X_start[:,0]=X_o

# fix the rhs of your ode and the runge-kutta time step in the solver using partial function
#Fixed jacobian in between observation gap
my_solver=jit(partial(solver,rhs_function=model_plus_jacobian,time_step=dt_solver))

#Store local growth rates, the G, R and C matrices
local_growth_rates=np.zeros((n1,num_clv))
Store_G=np.zeros((n1,model_dim,num_clv))
traj=np.zeros((n1+1,model_dim))
traj[0]=X_o

for i in range(n1):
    for j in range(n_iters):
        X_stop=my_solver(x_initial=X_start)
        X_start=X_stop
    #Compute the exponents and store the matrices
    traj[i+1]=X_start[:,0]
    local_growth_rates[i]=np.log(norm(X_stop[:,1:],axis=0))/dt
    X_start,Store_G[i],_=Output_tangent_QR(X_start)

os.mkdir('tf={}_dt={}_dt_solver={}_num_clv={}'.format(n1*dt,dt,dt_solver,num_clv))
os.chdir('tf={}_dt={}_dt_solver={}_num_clv={}'.format(n1*dt,dt,dt_solver,num_clv))
np.save('BLV.npy',Store_G)
np.save('exponents.npy',local_growth_rates)
np.save('traj.npy',traj)

print('Job done')


