"""Jaxed version of enkf.
The aim to to create a filter object"""

from jax import partial,jvp,vmap,jit,jacfwd 
import jax.numpy as jnp
import numpy as np
import json
import os
from numpy.linalg import norm,inv
from sklearn import ensemble
from ode_solvers import rk4_solver as solver
from testbed_models import L63
mvr=np.random.multivariate_normal

# All the assimilation setup make a filter object with all the details as attributed.

# use the model imported above
model_dim=3
num_clv=3

#model parameters if needed
rho = 28.0
sigma = 10.0
beta = 8.0/3

#fixing the parameters in your model's rhs, if it has any
model=partial(L63,sigma=10.0,rho=28.0,beta=8./3)

parameters={}
# Parameters to be controlled from here...
parameters['observables']=1
parameters['obs_gap']=0.1
parameters['solver_time_step']=0.01
parameters['mu']     =1.0
parameters['N']      =40
parameters['alpha']  =1.1              
parameters['loc']    ='False'
parameters['loc_fun']='none'
parameters['l_scale']=0
parameters['num_assimilations']=50000

#assign the parameters for the assimilation to variables
m=parameters['observables']
n=parameters['dim']
H=np.eye(model_dim)
delta_t=parameters['obs_gap']
dt_solver=parameters['solver_time_step']
mu=parameters['mu']
iters_delta_t=int(delta_t/dt_solver)
R= mu*np.eye(model_dim)
alpha=parameters['alpha']
ensemble_size=parameters['N']
num_assimilations=parameters['num_assimilations']
observations=np.load('')
Initial_ensemble=np.load('')

# the above model must be vectorized so that we can give multiple initial conditions
@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    Y_= vmap(model,in_axes=(0),out_axes=(0))(X_) 
    return Y_

my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

@jit
def forecast(last_analysis_ensemble):
    for i in range(iters_delta_t):
        forecast_ensemble=my_solver(last_analysis_ensemble)
    return forecast_ensemble

@jit
def analysis(forecast_ensemble,perturbed_obs,alpha):
    Cov_f_H_T=jnp.cov(alpha*forecast_ensemble)@np.transpose(H)
    Kalman_gain=Cov_f_H_T@jnp.inv(H@Cov_f_H_T+R)
    innovations=H@forecast_ensemble - perturbed_obs 
    analysis_ensemble=forecast_ensemble+Kalman_gain@innovations
    return analysis_ensemble

# We create a stamp for this filter run using model detaisl and all the parameters above 

# Allocate arrays for storing the ensembles
store_predicted_ensemble=np.zeros((num_assimilations,model_dim,ensemble_size))
store_filtered_ensemble=np.zeros_like(store_predicted_ensemble)

# Running enkf below
ensemble=Initial_ensemble
for i in range(num_assimilations):
    ensemble_predicted=forecast(ensemble)
    ensemble_observation=mvr(observations[i],R,ensemble_size).T
    ensemble_filtered=analysis(ensemble_predicted,ensemble_observation)
    store_predicted_ensemble[i]=ensemble_predicted
    store_filtered_ensemble[i]=ensemble_filtered

        
label='obs={}_ens={}_Mcov={},ocov={}_,gap={}_alpha={}_loc={}_r={}'.format(parameters['observables'],parameters['N'],parameters['lambda'],parameters['mu'],parameters['obs_gap'],parameters['alpha'],parameters['loc_fun'],parameters['l_scale'])
print('Job done, storing the ensembles now')
os.mkdir(label)
os.chdir(label)
np.save('filtered_ens.npy',store_filtered_ensemble)
np.save('predicted_ens.npy',store_predicted_ensemble)
print('finished')


