import numpy as np
import os
import json

"""Loading parameters and initial condition for lorenz_63"""
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/codes/assimilation')
parametersFile='L_63_x0_seed_{}.json'.format(45)
with open(parametersFile) as jsonfile:
     parameters=json.load(jsonfile)

# parameters={}
# parameters['initial']=np.load('hidden_path.npy')[0]
parameters['dim_x']=len(parameters['initial'])
mean_=parameters['initial']
#parameters['lambda_']=0.01

#initial bias in the ensemble........
biases=np.array([6.0])

#initial ensemble spread
covs_=np.array([4.0])
#Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])

# Finalized experiments in April,2021
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
#os.mkdir('ensembles')
os.chdir(os.path.join(os.getcwd(),'ensembles'))
for i in range(len(biases)):
    bias=biases[i]
    for j in range(len(covs_)):
        "Select seeds such that there is a unique value for each set of complete parameters"
        seed_num=int(47*(i+1)*(j+1))
        np.random.seed(seed_num)
        parameters['lambda_']=covs_[j]
        Initial_cov=parameters['lambda_']*np.eye(parameters['dim_x'])
        x0_ensemble=np.random.multivariate_normal(np.asarray(mean_)+bias,Initial_cov,50).T
        np.save('Ensemble={}_bias_{}_init_cov_{}_seed_{}.npy'.format(50,bias,covs_[j],seed_num),x0_ensemble)
        #np.save('Ensemble={}_init_cov_{}_seed_{}.npy'.format(i,parameters['lambda_'],seed_num),x0_ensemble)

print('Job Done')

