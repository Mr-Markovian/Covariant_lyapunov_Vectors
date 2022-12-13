"""Jaxed version of enkf.
The aim to to perform DA experiment by performing forecast and analysis steps. The 
initial ensemble and the observations(all time) are loaded. All important and repititive parts 
of the code are jitted"""

from jax import partial,vmap,jit
import jax.numpy as jnp
import numpy as np
import os
from ode_solvers import rk4_solver as solver
from testbed_models import L96
mvr=np.random.multivariate_normal

# use the model imported above
model_dim=40
num_clv=model_dim

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
parameters['ebias']=4.0
parameters['ecov']=2.0
parameters['observables']=40
parameters['model_dim']=40
parameters['obs_gap']=0.05
parameters['solver_time_step']=0.01
parameters['mu']     =1.0
parameters['N']      =20
parameters['alpha']  =1.0           
parameters['loc']    ='False'
parameters['loc_fun']='none'
parameters['l_scale']=0
parameters['num_assimilations']=12000

#assign the parameters for the assimilation to variables
ebias=parameters['ebias']
ecov=parameters['ecov']
m=parameters['observables']
n=parameters['model_dim']
l_scale=parameters['l_scale']
delta_t=parameters['obs_gap']
dt_solver=parameters['solver_time_step']
mu=parameters['mu']
iters_delta_t=int(delta_t/dt_solver)

alpha=parameters['alpha']
ensemble_size=parameters['N']
num_assimilations=parameters['num_assimilations']

#Obervation operator for alternate coordinates 
# H=np.zeros((m,n))
# for i in range(m):
#     H[i,2*i]=1


#Obervation operator for full state 
H=np.eye(n)


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


# the above model must be vectorized so that we can give multiple initial conditions along columns
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
k=4
exp_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-40/seed_3_obs_full_state'
os.chdir(exp_path)

#os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/assimilation')
# Change to the path where observations and the ensemble are present

Initial_ensemble=np.load('ensembles/Ensemble=50_bias_{}_init_cov_{}_seed_47.npy'.format(ebias,ecov))[:,:ensemble_size]

#The value of observation error standard deviation in each component is mu
#mus=np.array([0.1,0.3,0.5,0.7,0.9])
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
for sigma in sigmas:
    os.chdir(exp_path)
    parameters['mu']=np.round(sigma**2,2)
    R= np.eye(m) ** sigma**2
    observations=np.load('sigma={}/state_noisy_g=0.05_sigma={}.npy'.format(sigma,sigma))

    #load observations
    #os.chdir(exp_path+'/ob{}/'.format(k))
    #observations=np.load('ob{}_gap_0.05_H2__mu={}_obs_cov1.npy'.format(k,mu))
    
    store_filtered_ensemble=np.zeros((num_assimilations,model_dim,ensemble_size))

    # Running enkf below
    ensemble=Initial_ensemble
    store_filtered_ensemble[0]=Initial_ensemble
    for i in range(1,num_assimilations):
        ensemble_predicted=forecast(ensemble)
        ensemble_observation=mvr(observations[i],R,ensemble_size).T   
        ensemble_filtered=analysis(ensemble_predicted,ensemble_observation,alpha)
        store_filtered_ensemble[i]=ensemble_filtered
        ensemble=ensemble_filtered
    
    label='ebias={}_ecov={}_obs={}_ens={}_mu={}_gap={}_alpha={}_loc={}_r={}'.format(ebias, ecov,parameters['observables'],parameters['N'],parameters['mu'],parameters['obs_gap'],parameters['alpha'],parameters['loc_fun'],parameters['l_scale'])
    print('Job done, storing the ensembles now')
    os.mkdir(label)
    os.chdir(label)
    np.save('filtered_ens.npy',store_filtered_ensemble)
print('All done')


