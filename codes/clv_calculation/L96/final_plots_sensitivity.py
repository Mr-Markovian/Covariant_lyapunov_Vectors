from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['lines.markersize']=10
mpl.rcParams['axes.titlesize']=30
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=20

dt=0.05   
dt_solver=0.01
model_dim=40
model_name='L96'
num_clv=model_dim
seed_num=3

spr=1
#data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/seed_{}'.format(model_dim,seed_num)
data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L96/L96-{}/explanation'.format(model_dim)

os.chdir(data_path)
sigmas=np.array([0.1,0.2,0.3,0.4,0.5])
plt.figure(figsize=(16,8))
for sigma in sigmas:
    # clv_cosines=np.load('clv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    # plt.plot(np.arange(1,num_clv+1),np.mean(clv_cosines,axis=0))
    # plt.errorbar(x=np.arange(1,num_clv+1),y=np.mean(clv_cosines,axis=0),yerr=np.std(clv_cosines,axis=0),capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)

    clv_cosines=np.load('blv_cosine_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
    quantiles_=np.quantile(clv_cosines,[0.25,0.50,0.75],axis=0)
    asymmetric_bar=np.zeros((2,num_clv))
    asymmetric_bar[0]=quantiles_[1]-quantiles_[0]
    asymmetric_bar[1]=quantiles_[2]-quantiles_[1]
    #plt.plot(np.arange(1,num_clv+1),np.mean(blv_cosines,axis=0))
    plt.errorbar(x=np.arange(1,num_clv+1),y=np.median(clv_cosines,axis=0),yerr=asymmetric_bar,capsize=4,label=r'$\sigma$={}'.format(sigma),marker='.',ms=20)

plt.xlabel(r'index $\to$')
plt.ylim(0,1.1)
#plt.yticks(np.arange(0.3,1.1,0.1))
plt.xticks(np.arange(1,num_clv+1))
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel(r'cosine $(\theta)$')
plt.legend(loc='lower center',ncol=len(sigmas))
#plt.savefig('clv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))
plt.savefig('blv_angle_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))

# plt.figure(figsize=(16,8))
# for sigma in sigmas:
#     lexp=np.load('exp_sigma={}_model_dim={}_num_cl={}.npy'.format(sigma,model_dim,num_clv))
#     plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(sigma),s=20)

# plt.scatter(np.arange(1,num_clv+1),lexp,label=r'$\sigma$={}'.format(0.0),s=20,c='black')
# plt.xlabel(r'index $\to$')
# plt.ylabel(r'$\hat \lambda $')
# plt.xticks(np.arange(1,num_clv+1))
# plt.legend()
# plt.savefig('exponent_sensitivity_L96-{}_seed_{}.png'.format(model_dim,seed_num))


