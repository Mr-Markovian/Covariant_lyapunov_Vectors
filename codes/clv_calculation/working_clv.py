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
from numpy.linalg import norm,inv,qr
from scipy.integrate import solve_ivp 
import json 
from testbed_models import L96,L63

qr_decomp=jnp.linalg.qr
diagonal=np.diag
seed_num=41
np.random.seed(seed_num)

os.chdir('/home/shashank/Documents/Data Assimilation/PDE_DA/CLV/L63')
#Model Details.....................................................................
# model_name='Lorenz96'
# if model_name=='Lorenz96':
#     print('Model Name:',model_name,'dimension='model_dim,'forcing='forcing)
model=L63
model_dim=3
#forcing=8
num_clv=3
print('To calculate first {} covariant lyapunov vectors '.format(num_clv))

#Model details for L63
#sigma,rho,beta=10.,28.,8./3

# time step for reorthonormalsation
delta_t1=0.01
delta_t2=0.001

# Trajectory details 
f_transient_i=0
f_transient_f=1000
interval_i=1000
interval_f=1010
b_transient_i=1010
b_transient_f=1500           

T1=np.arange(f_transient_i,f_transient_f+delta_t1,delta_t1)
T2=np.arange(interval_i,interval_f+delta_t2,delta_t2)
T3=np.arange(b_transient_i,b_transient_f+delta_t2,delta_t2)

# Number of time steps in these intervals
n1,n2,n3=len(T1),len(T2),len(T3)

# T2 is going to be used twice, first during forward run, second during sampling the clvs 

""" Choice of initial vectors here is standard orthonormal vectors """
# with open('L_96_I_x0_on_attractor_dim_40,forcing_8.json') as jsonfile:
#     param=json.load(jsonfile)

#Initial condition of the transient trajectory
X_start=np.zeros((num_clv+1,model_dim))

#X_start[0]=np.asarray(param['initial'])
X_start[0]=np.random.random(model_dim)
X_start[1:]=np.eye(model_dim)[:num_clv]  # orthogonal perturbations

#Store local growth rates
local_growth_rates=np.zeros((len(T1)+len(T2)+len(T3)-1,num_clv))

#Store Q and R matrices
Store_G=np.zeros((len(T2)-1,model_dim,num_clv))
Store_R=np.zeros((len(T2)+len(T3)-2,num_clv,num_clv))
Store_C=np.zeros((len(T2)-1,num_clv,num_clv))

def fun(t,x):
    "RHS for scipy.integrate.solve_ivp"
    return np.asarray(model_plus_jacobian(x))

@jit
def model_plus_jacobian(x_):
    "model + jacobian for state space and tanegent linear evolution at a point"
    X_=x_.reshape((num_clv+1,model_dim))
    x_eval=X_[0]  # Nx1
    in_tangents=X_[1:,:]
    # The second argument below will take the RHS of dynamical equation.
    pushfwd = partial(jvp, jit(model) , (x_eval,))
    y, out_tangents = vmap(pushfwd,in_axes=(0), out_axes=(None,0))((in_tangents,)) 
    return jnp.concatenate((y,out_tangents.ravel()))

@jit
def Output_tangent_QR(x_):
    """Split the vectors for the trajectory point and the tangents and store 
    the vectors and the upper triangular matrix Q and R respectively"""
    Q_,R_=qr_decomp(jnp.transpose(x_[1:,:]))  # QR decomposition
    #cn=np.linalg.cond(jnp.transpose(temp_[1:,:])) 
    #local_growth_rates_= np.sum(np.absolute(R_),axis=1), should give same ans
    temp_=x_.at[1:].set(jnp.transpose(Q_))
    return temp_,Q_,R_

def backward_evolution(x_,y_k):
    """Generates coeffients at the previous time step from the current time step"""
    temp_=inv(x_)@y_k
    #row_sums = temp_.sum(axis=1,keepdims=True)
    row_sums=np.linalg.norm(temp_,ord=2,axis=0,keepdims=True)
    temp_ = temp_ / row_sums
    return temp_

# For the initial transient, we now integrate this,Storing the trajectory and the exponents:
for j in range(n1-1):
    soln=solve_ivp(fun,(T1[j],T1[j+1]),X_start.flatten())
    X_stop=soln.y[:,-1].reshape(num_clv+1,model_dim)
    #Compute the exponents 
    local_growth_rates[j]=np.log(norm(X_stop[1:],axis=1))/delta_t1
    X_start,_,_=Output_tangent_QR(X_stop)

print('Transient is Over, moving on to the interval in question')
# For the interval we want to sample over
# Store Q and R

trajectory=[]
for j in range(n2-1):
    soln=solve_ivp(fun,(T2[j],T2[j+1]),X_start.flatten())
    X_stop=soln.y[:,-1].reshape(num_clv+1,model_dim)
    trajectory.append(X_stop[0])
    #Compute the exponents and store the matrices
    local_growth_rates[n1-1+j]=np.log(norm(X_stop[1:],axis=1))/delta_t2
    X_start,Store_G[j],Store_R[j]=Output_tangent_QR(X_stop)

#Free Up space
np.save('trajectory_{}_seed_{}.npy'.format(model_dim,seed_num),np.asarray(trajectory))
np.save('matrices_g_{}_seed_{}.npy'.format(model_dim,seed_num),Store_G)  
del Store_G  

print('The Interval is Over, moving forward on the backward transient part')    
# Going forward over the backward transient interval
# Store Q and R
for j in range(n3-1):
    soln=solve_ivp(fun,(T3[j],T3[j+1]),X_start.flatten())
    X_stop=soln.y[:,-1].reshape(num_clv+1,model_dim)
    #Compute the exponents and store the matrices
    local_growth_rates[n1+n2-2+j]=np.log(norm(X_stop[1:],axis=1))/delta_t2
    X_start,_,Store_R[n2-1+j]=Output_tangent_QR(X_stop)

# Free up space
np.save('local_growth_rates_{}.npy'.format(model_dim),local_growth_rates)
del local_growth_rates

print('Now, Moving backward on the backward transient part')    
#Creating a generic upper triangular matirx
C_tilde_initial=np.zeros((num_clv,num_clv))
for i in range(num_clv):
    C_tilde_initial[i,i:]=np.ones(num_clv-i)
C_tilde=C_tilde_initial

# Now going backwards over the backward transient interval, using backward evolution  
# Use R-inverse stored earlier
for j in range(n3-1):
    l=n2+n3-3-j
    C_tilde=backward_evolution(Store_R[l],C_tilde)

print(' Ready to sample clvs over the interval')
# Now sampling the clvs(i.e. in terms of coefficients when expressed in gs vs )
# Use R-inverse stored earlier
for j in range(n2-1):
    l=n2-2-j
    C_tilde=backward_evolution(Store_R[l],C_tilde)
    Store_C[l]=C_tilde

print('Sampled, now storing all data')

# Save both the initial condition and the forward gram-schmidt vectors:
#np.save('matrices_r_{}.npy'.format(model_dim),Store_R)
np.save('matrices_c_{}_seed_{}.npy'.format(model_dim,seed_num),Store_C)
print('Job done')
