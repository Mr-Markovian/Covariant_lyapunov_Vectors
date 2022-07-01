"""This script implements the Ensemble kalamn Filter algorithm by defining the class Enkf
which inherits the model class of the respective system(ode or pde)."""

"""The Five crucial sub-parts of the code are:
1)generate: Generate the initial ensemble at time t=0
2)forecast: Integrate the ensemble over the observation time gap (defined as obs_gap)
3)observation: Integrate to get the state and the observation and observation perturbations
4)analysis : Analyse once the observations are available
5)simulate: Calls all the above functions to perform the respective tasks"""

#Later parts of the code has options to plot various things and were used
#while writing the code in jupyter-notebook.Not much useful once we start saving and visualizing data

from lorenz_63_ode import *
import numpy as np

# A bit of speeding up...
zo=np.zeros
zo_l=np.zeros_like
mvr=np.random.multivariate_normal
ccat=np.concatenate
p_inv=np.linalg.pinv
abs=np.absolute

class Enkf(lorenz_63):
    def __init__(self,parameters):
        """Class attributes definition and initialization"""
        # parse parameters
        for key in parameters:
            setattr(self, key, parameters[key])
        self.final        =self.initial
        self.forecast_mean=zo((1,self.dim))                # To store the predicted value of the state
        self.analysis_mean=zo((1,self.dim))                # to store the estimate of the true State
        self.analysis_ensemble=zo((1,self.dim,self.N))     # to store the analyized Ensemble
        self.forecast_ensemble=zo((1,self.dim,self.N))     # to store the forecast Ensemble
        self.innov_   =zo((1,self.observables,self.N))     # storing the innovation vector
        self.A_p      =zo((self.dim,self.N))
        self.A        =self.initial_ensemble
        self.D_p      =zo((self.observables,self.N))
        self.analysis_mean[0]    =self.initial
        self.forecast_mean[0]    =self.initial
        self.analysis_ensemble[0]=self.initial_ensemble
        self.forecast_ensemble[0]=self.initial_ensemble
        self.time=self.obs_gap*np.arange(0,self.assimilations)

        # The label created by using the most important set of parameters used to create the Object
        self.label    ='obs={}_ens={}_Mcov={},ocov={}_,gap={}_alpha={}_loc={}_r={}'.format(parameters['observables'],parameters['N'],parameters['lambda'],parameters['mu'],parameters['obs_gap'],parameters['alpha'],parameters['loc_fun'],parameters['l_scale'])
        # if localization is implemented,create the localization matrix 'self.rho'
        if (self.loc=='True'):
            if (self.loc_fun=='convex'):
                f1=lambda r:0 if r>self.l_scale-1 else np.round(2*np.exp(-r/self.l_scale)/(1+np.exp(r)),3)
                self.rho=np.asarray([[f1(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])
            if (self.loc_fun=='concave'):
                f2=lambda r:0 if r>self.l_scale else np.round((self.l_scale**2-r*r)/self.l_scale**2,3) # Concave
                self.rho=np.asarray([[f2(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])
            if (self.loc_fun=='flat'):
                f3=lambda r:0 if r>self.l_scale else np.round((self.l_scale-r)/self.l_scale,3)
                self.rho=np.asarray([[f3(min(abs(i-j),abs(self.dim-abs(i-j)))) for j in range(self.dim)]for i in range(self.dim)])

    def forecast(self):
        "Integrate the ensemble until new observation"
        "Save the forcast upto initial time+obs_gap"
        self.time_start=self.time[self.i]
        self.time_stop=self.time[self.i+1]
        self.t_evaluate=None
        for i in range(self.N):
            self.initial   =self.A[:,i]
            self.A[:,i]    =self.solution()[-1]

        self.forecast_ensemble=ccat((self.forecast_ensemble,[self.A]),axis=0)
        mean                  =np.sum(self.A,axis=1)/(self.N)
        self.forecast_mean    =ccat((self.forecast_mean,[mean]),axis=0)
        # To compute and the ensemble perturbations and use multiplicative inflation
        self.A_p              =self.alpha*(self.A-np.tile(mean,(self.N,1)).T)

    def analysis(self):
        """All computation for analysis after observation"""

        obs_t=self.obs[self.i+1]
        self.D           =mvr(obs_t,self.obs_cov,self.N).T
        temp             =np.sum(self.D,axis=1)/(self.N)  # temporary variable to store the mean

        # To compute and store innovation vectors
        self.innovation   =self.D-self.H@self.A

        """use k_gain=(P_f)(H)^t[H P_f H^t + R ]^(-1) if localization is implemented"""
        p_f  =self.A_p@np.transpose(self.A_p)/(self.N-1)
        if (self.loc=='True'):   # If localization is implemented,replace p_f by schur product of self.rho and p_f
            p_f =self.rho*p_f
        p_f_h_t =p_f@np.transpose(self.H)

        # Compute the kalman gain matrix:
        k_gain=p_f_h_t@p_inv(self.H@p_f_h_t+self.obs_cov)

        # The analysis ensemble
        self.A            =self.A+k_gain@self.innovation

        # The analysis mean
        mean              =np.sum(self.A,axis=1)/(self.N)
        self.innov_       =ccat((self.innov_,[self.innovation]),axis=0)
        self.analysis_mean=ccat((self.analysis_mean,[mean]),axis=0)

        # Storing the analysis ensemble at current time
        self.analysis_ensemble=ccat((self.analysis_ensemble,[self.A]),axis=0)

    def simulate(self):
        """Time-total is number of Assimilations performed"""
        for self.i in range(self.assimilations-1):
            self.forecast()
            self.analysis()

    def save_data(self):
        "to save forecast,analysis ensemble with time"
        np.save(self.label+'f_ensemble.npy',self.forecast_ensemble)
        np.save(self.label+'a_ensemble.npy',self.analysis_ensemble)
        np.save(self.label+'time.npy',self.time)
