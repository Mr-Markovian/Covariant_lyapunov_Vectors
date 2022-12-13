"""Jaxed version of enkf.
The aim to to perform DA experiment by performing forecast and analysis steps. The 
initial ensemble and the observations(all time) are loaded. All important and repititive parts 
of the code are jitted"""

from jax import partial,vmap,jit
import jax.numpy as jnp
import numpy as np
import os
from ode_solvers import rk4_solver as solver
from testbed_models import L63,L96
mvr=np.random.multivariate_normal

# use the model imported above
model_dim=40
num_clv=40

#model parameters if needed
#rho = 28.0
#sigma = 10.0
#beta = 8.0/3

#model parameter for L96
forcing=8.0

#fixing the parameters in your model's rhs, if it has any
model=partial(L96,forcing=8.0)

parameters={}
# Parameters to be controlled from here...
parameters['ebias']=6.0
parameters['ecov']=4.0
parameters['observables']=1
parameters['model_dim']=model_dim
parameters['obs_gap']=0.05
parameters['solver_time_step']=0.01
parameters['mu']     =1.0
parameters['N']      =25
parameters['alpha']  =1.0             
parameters['loc']    ='True'
parameters['loc_fun']='gaspri'
parameters['l_scale']=4
parameters['num_assimilations']=50000

#assign the parameters for the assimilation to variables
ebias=parameters['ebias']
ecov=parameters['ecov']
m=parameters['observables']
n=parameters['model_dim']
l_scale=parameters['l_scale']
#Observation Operator for observing y 
H=np.zeros((m,n))
H[0,1]=1   
delta_t=parameters['obs_gap']
dt_solver=parameters['solver_time_step']
mu=parameters['mu']
iters_delta_t=int(delta_t/dt_solver)
R= mu*np.eye(m)
alpha=parameters['alpha']
ensemble_size=parameters['N']
num_assimilations=parameters['num_assimilations']

# defining the localization matrix in covariance localization scheme:
def gaspri(r):
    #Defining the function as given in Alberto-Carassi DA-review paper
    r_=abs(r)/l_scale
    if 0.<=r_ and r_<1.:
        return 1.-(5./3)*r_**2+(5./8)*r_**3+(1./2)*r_**4-(1./4)*r_**5
    elif 1.<=r_ and r_<2.:
        return 4.-5.*r_+(5./3.)*r_**2+(5./8.)*r_**3-(1./2.)*r_**4+(1/12)*r_**5-2/(3*r_)
    else:
        return 0

if parameters['loc']=='True':
    rho=np.asarray([[gaspri(min(abs(i-j),abs(n-abs(i-j)))) for j in range(n)]for i in range(n)])
else :
    rho=np.eye(n)

os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
# Change to the path where observations and the ensemble are present
observations=np.load('ob2/ob2_gap_0.01_H1__mu=4.0_obs_cov1.npy')
Initial_ensemble=np.load('ensembles/Ensemble=50_bias_{}_init_cov_{}_seed_47.npy'.format(ebias,ecov))[:,:ensemble_size]

# the above model must be vectorized so that we can give multiple initial conditions
@jit
def model_vectorized(X_):
    "model vectorization using vmap, we can compute multiple forward passes"
    # The second argument below will take the RHS of dynamical equation.
    Y_= vmap(model,in_axes=(1),out_axes=(1))(X_) 
    return Y_

my_solver=jit(partial(solver,rhs_function=model_vectorized,time_step=dt_solver))

@jit
def forecast(last_analysis_ensemble):
    #forecast_ensemble=lax.fori_loop(0,iters_delta_t,my_solver,last_analysis_ensemble)
    forecast_ensemble=last_analysis_ensemble
    for i in range(iters_delta_t):
        forecast_ensemble=my_solver(x_initial=forecast_ensemble)
    return forecast_ensemble

@jit
def analysis(forecast_ensemble,perturbed_obs,alpha):
    P_f_H_T=(rho*jnp.cov(alpha*forecast_ensemble,rowvar=True))@jnp.transpose(H)
    Kalman_gain=P_f_H_T@jnp.linalg.inv(H@P_f_H_T+R)
    innovations=perturbed_obs-H@forecast_ensemble
    analysis_ensemble=forecast_ensemble+(Kalman_gain@innovations)
    return analysis_ensemble

# We create a stamp for this filter run using model detaisl and all the parameters above 

# Allocate arrays for storing the ensembles
store_predicted_ensemble=np.zeros((num_assimilations,model_dim,ensemble_size))
store_filtered_ensemble=np.zeros_like(store_predicted_ensemble)

# Running enkf below
ensemble=Initial_ensemble
for i in range(num_assimilations-1):
    ensemble_predicted=forecast(ensemble)
    ensemble_observation=mvr(observations[i+1],R,ensemble_size).T
    ensemble_filtered=analysis(ensemble_predicted,ensemble_observation,alpha)
    store_predicted_ensemble[i]=ensemble_predicted
    store_filtered_ensemble[i]=ensemble_filtered
    ensemble=ensemble_filtered

label='ebias={}_ecov={}_obs={}_ens={}_ocov={}_gap={}_alpha={}_loc={}_r={}'.format(ebias, ecov,parameters['observables'],parameters['N'],parameters['mu'],parameters['obs_gap'],parameters['alpha'],parameters['loc_fun'],parameters['l_scale'])
print('Job done, storing the ensembles now')
print(store_filtered_ensemble.shape)

os.mkdir(label)
os.chdir(label)
np.save('filtered_ens.npy',store_filtered_ensemble)
np.save('predicted_ens.npy',store_predicted_ensemble)
print('Job done')


