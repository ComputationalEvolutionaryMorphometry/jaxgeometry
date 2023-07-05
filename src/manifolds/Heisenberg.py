## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#


from src.setup import *
from src.params import *

from src.manifolds.manifold import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class Heisenberg(Manifold):
    """ Heisenberg group """

    def __init__(self,):
        Manifold.__init__(self)

        self.dim = 3
        self.sR_dim = 2

        self.update_coords = lambda coords,chart: (coords[0],chart)

        ##### (orthonormal) distribution
        self.D = lambda x: jnp.array([[1,0,-x[0][1]/2],[0,1,x[0][0]/2]]).T
        #self.D = lambda x: jnp.array([[1,0,0],[0,1,0],[0,0,1]]).T

    def __str__(self):
        return "Heisenberg group"

    def plot(self):
        None

    def plot_path(self, xs, vs=None, v_steps=None, i0=0, color='b', 
                  color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
    
        if vs is not None and v_steps is not None:
            v_steps = np.arange(0,n_steps)
    
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, v=vs[i] if vs is not None else None,
                       v_steps=v_steps,i=i,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==(N-1))
            prevx = x 
        return

    # plot x in coordinates
    def plotx(self, x, u=None, v=None, v_steps=None, i=0, color='b',               
              color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
        if (type(x) != type(())):
            x = (x,)
        if (prevx is not None and type(prevx) != type(())):
            prevx = (prevx,)
    
        if v is not None and v_steps is None:
            v_steps = np.arange(0,n_steps,10)        
    
        ax = plt.gca()
        if prevx is None or last:
            ax.scatter(x[0][0],x[0][1],x[0][2],color=color,s=s)
        if prevx is not None:
            xx = np.stack((prevx[0],x[0]))
            ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
    
        if u is not None:
            ax.quiver(x[0][0], x[0][1], x[0][2], u[0], u[1], u[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                      color='black')
    
        if v is not None:
            if i in v_steps:
                ax.quiver(x[0][0], x[0][1], x[0][2], v[0], v[1], v[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=1.0,
                        color='black')

# funtion to evaluate spherical harmonics
def Y(l,m,theta,phi):
    if l == 0 and m == 0:
        return 1/np.sqrt(4*np.pi)
    elif l == 1 and m == -1:
        return np.sqrt(3/(4*np.pi))*np.sin(theta)*np.exp(1j*phi)
    elif l == 1 and m == 0:
        return np.sqrt(3/(4*np.pi))*np.cos(theta)
    elif l == 1 and m == 1:
        return np.sqrt(3/(4*np.pi))*np.sin(theta)*np.exp(-1j*phi)
    elif l == 2 and m == -2:
        return np.sqrt(15/(16*np.pi))*np.sin(theta)**2*np.exp(2j*phi)
    elif l == 2 and m == -1:
        return np.sqrt(15/(8*np.pi))*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
    elif l == 2 and m == 0:
        return np.sqrt(5/(16*np.pi))*(3*np.cos(theta)**2-1)
    elif l == 2 and m == 1:
        return np.sqrt(15/(8*np.pi))*np.sin(theta)*np.cos(theta)*np.exp(-1j*phi)
    elif l == 2 and m == 2:
        return np.sqrt(15/(16*np.pi))*np.sin(theta)**2*np.exp(-2j*phi)
    else:
        return 0
    
