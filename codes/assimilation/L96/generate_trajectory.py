from lorenz_96_ode import *
#from II_scale_lorenz_96 import *
import numpy as np
import json
seed_num=35
import os
np.random.seed(seed_num)

"""Loading parameters and initial condition for lorenz_96"""
parametersFile='L_96_I_x0_on_attractor_dim_40,forcing_8.json'
with open(parametersFile) as jsonfile:
     parameters=json.load(jsonfile)

#Total time of the trajectory
parameters['time_start']=0
parameters['time_stop']=500

parameters['obs_gap']=0.05
#integrate to generate the trajectory:
parameters['t_evaluate']=np.arange(0,parameters['time_stop'],parameters['obs_gap'])
ob_=lorenz_96(parameters)
os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96')
np.save('Trajectory_g={}_seed_{}.npy'.format(parameters['obs_gap'],seed_num),ob_.solution())
print('Job Done')
