import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data')
state=np.load('Trajectory_{}_T={}.npy'.format(0.01,500))

fig=plt.figure(figsize=(16,8))

ax=fig.add_subplot(projection ='3d')
ax.plot3D(state[:,0], state[:,1], state[:,2],label='trajectory')
ax.set_title('Trajectory',fontsize=18)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



