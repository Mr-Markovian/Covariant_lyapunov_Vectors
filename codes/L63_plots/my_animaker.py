from mayavi.mlab import *
from mayavi import mlab
import numpy as np
import os


data_path='/home/shashank/Documents/Data Assimilation/ENKF_for_CLVs/data/L63/seed_3'

spr=2
model_dim=3
model_name='L63'
base_type='state_noisy'
num_clv=3
coord=['X','Y','Z']
dt=0.01
mu=1.0
t_start=0
t_stop=5000
start_idx=10000


# vectors for the true trajectory
os.chdir(data_path+'/sigma={}'.format(0.0))
C=np.load('matrices_c_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
G=np.load('matrices_g_{}_model_{}_{}.npy'.format(model_dim,model_name,base_type))[t_start:t_stop]
base_traj=np.load('{}_g={}_sigma={}.npy'.format(base_type,dt,0.0))[start_idx+t_start:start_idx+t_stop]

V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]

def shorten(arr, factor=5):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

clv_index=0
x,y,z=shorten(base_traj[:,0]),shorten(base_traj[:,1]),shorten(base_traj[:,2])
u,v,w=shorten(V[:,0,clv_index]),shorten(V[:,1,clv_index]),shorten(V[:,2,clv_index])
#u1,v1,w1=shorten(V[:,clv_index+1,0]),shorten(V[:,clv_index+1,1]),shorten(V[:,clv_index+1,2])
mlab.figure(bgcolor=(0, 0, 0))
my_field=mlab.quiver3d(x,y,z,u,v,w,color=(0.5,0.5,0.5))
#my_field=mlab.quiver3d(x,y,z,u1,v1,w1,color=(0.1,0.5,0.1))

coord_lims=(-20,20,-30,30,0,50)
mlab.axes(my_field, color=(.7, .7, .7), extent=coord_lims, xlabel='x', ylabel='y',zlabel='z')
mlab.outline(my_field, color=(.3, .3, .3), extent=coord_lims)
mlab.text(0, 0, 'CLV {}'.format(clv_index+1), z=-4, width=0.5)
mlab.show()



# Adding a fixed coordinate system using outline

# Adding objects at different time and animating 40 random walks.
