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

import matplotlib.pyplot as plt

class Manifold(object):
    """ Base manifold class """

    def __init__(self):
        self.dim = None        
        self.rank = None
        if not hasattr(self, 'do_chart_update'):
            self.do_chart_update = None # set to relevant function if what updates are desired

    def chart(self):
        """ return default or specified coordinate chart. This method will generally be overriding by inheriting classes """
        # default value 
        return jnp.zeros(1) 

    def centered_chart(self,coords):
        """ return centered coordinate chart. Must be implemented by inheriting classes """
        return jnp.zeros(1)

    def coords(self,coords=None,chart=None):
        """ return coordinate representation of point in manifold """
        if coords is None:
            coords = jnp.zeros(self.dim)
        if chart is None:
            chart = self.chart()

        return (jnp.array(coords),chart)

    def update_coords(self,coords,new_chart):
        """ change between charts """
        assert(False) # not implemented here

    def update_vector(self,coords,new_coords,new_chart,v):
        """ change tangent vector between charts """
        assert(False) # not implemented here

    def update_covector(self,coords,new_coords,new_chart,p):
        """ change cotangent vector between charts """
        assert(False) # not implemented here

    def newfig(self):
        """ open new plot for manifold """

    def __str__(self):
        return "abstract manifold"

class EmbeddedManifold(Manifold):
    """ Embedded manifold base class """

    def update_coords(self,coords,new_chart):
        """ change between charts """
        return (self.invF((self.F(coords),new_chart)),new_chart)

    def update_vector(self,coords,new_coords,new_chart,v):
        """ change tangent vector between charts """
        return jnp.dot(self.invJF((self.F((new_coords,new_chart)),new_chart)),jnp.dot(self.JF(coords),v))

    def update_covector(self,coords,new_coords,new_chart,p):
        """ change cotangent vector between charts """
        return jnp.dot(self.JF((new_coords,new_chart)).T,jnp.dot(self.invJF((self.F(coords),coords[1])).T,p))

    def __init__(self,F=None,dim=None,emb_dim=None,invF=None):
        Manifold.__init__(self)
        self.dim = dim
        self.emb_dim = emb_dim

        # embedding map and its inverse
        if F is not None:
            self.F = F
            self.invF = invF
            self.JF = jacfwdx(self.F)
            self.invJF = jacfwdx(self.invF)

            # metric matrix
            self.g = lambda x: jnp.dot(self.JF(x).T,self.JF(x))


    def plot_path(self, xs, vs=None, v_steps=None, i0=0, color='b', 
                  color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
    
        if vs is not None and v_steps is not None:
            v_steps = np.arange(0,n_steps)
    
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            xx = x[0] if type(x) is tuple else x
            if xx.shape[0] > self.dim and (self.emb_dim == None or xx.shape[0] != self.emb_dim): # attached vectors to display
                v = xx[self.dim:].reshape((self.dim,-1))
                x = (xx[0:self.dim],x[1]) if type(x) is tuple else xx[0:self.dim]
            elif vs is not None:
                v = vs[i]
            else:
                v = None
            self.plotx(x, v=v,
                       v_steps=v_steps,i=i,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==(N-1))
            prevx = x 
        return

    # plot x. x can be either in coordinates or in R^3
    def plotx(self, x, u=None, v=None, v_steps=None, i=0, color='b',               
              color_intensity=1., linewidth=1., s=15., prevx=None, prevchart=None, last=True):
    
        assert(type(x) == type(()) or x.shape[0] == self.emb_dim)
    
        if v is not None and v_steps is None:
            v_steps = np.arange(0,n_steps)        
    
        if type(x) == type(()): # map to manifold
            Fx = self.F(x)
            chart = x[1]
        else: # get coordinates
            Fx = x
            chart = self.centered_chart(Fx)
            x = (self.invF((Fx,chart)),chart)

        if prevx is not None:
            if type(prevx) == type(()): # map to manifold
                Fprevx = self.F(prevx)
            else:
                Fprevx = prevx
                prevx = (self.invF((Fprevx,chart)),chart)
    
        ax = plt.gca()
        if prevx is None or last:
            ax.scatter(Fx[0],Fx[1],Fx[2],color=color,s=s)
        if prevx is not None:
            xx = np.stack((Fprevx,Fx))
            ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
    
        if u is not None:
            Fu = np.dot(self.JF(x), u)
            ax.quiver(Fx[0], Fx[1], Fx[2], Fu[0], Fu[1], Fu[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                      color='black')
    
        if v is not None:
            if i in v_steps:
                v = np.dot(self.JF(x), v)
                ax.quiver(Fx[0], Fx[1], Fx[2], v[0], v[1], v[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                        color='black')

    def __str__(self):
        return "dim %d manifold embedded in R^%d" % (self.dim,self.emb_dim)
