from mayavi.mlab import *
from mayavi import mlab
import numpy as np
import os

os.chdir('/home/shashank/Documents/Data Assimilation/PDE_DA/CLV/L63')

model_dim=3
coord=['X','Y','Z']
traj=np.load('trajectory_{}.npy'.format(model_dim))
C=np.load('matrices_c_{}.npy'.format(model_dim))
G=np.load('matrices_g_{}.npy'.format(model_dim))
V=np.zeros_like(G)
for i in range(G.shape[0]):
    V[i]=G[i]@C[i]

def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

clv_index=0
x,y,z=shorten(traj[:,0]),shorten(traj[:,1]),shorten(traj[:,2])
u,v,w=shorten(V[:,clv_index,0]),shorten(V[:,clv_index,1]),shorten(V[:,clv_index,2])
u1,v1,w1=shorten(V[:,clv_index+1,0]),shorten(V[:,clv_index+1,1]),shorten(V[:,clv_index+1,2])
mlab.figure(bgcolor=(0, 0, 0))
my_field=mlab.quiver3d(x,y,z,u,v,w,color=(0.5,0.5,0.5))
my_field=mlab.quiver3d(x,y,z,u1,v1,w1,color=(0.1,0.5,0.1))

coord_lims=(-16,16,-25,25,0,50)
mlab.axes(my_field, color=(.7, .7, .7), extent=coord_lims, xlabel='x', ylabel='y',zlabel='z')
mlab.outline(my_field, color=(.3, .3, .3), extent=coord_lims)
mlab.text(0, 0, 'CLV {}'.format(clv_index+1), z=-4, width=0.5)
mlab.show()


# Adding a fixed coordinate system using outline

# Adding objects at different time and animating 40 random walks.
