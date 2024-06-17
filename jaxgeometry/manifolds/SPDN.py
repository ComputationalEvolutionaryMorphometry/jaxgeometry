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
# along with Jax Geometry. If not, see <http://www.gnu.org/licenses/>.
#

from jaxgeometry.setup import *
from jaxgeometry.params import *

from jaxgeometry.manifolds.manifold import *

from jaxgeometry.plotting import *
import matplotlib.pyplot as plt
from matplotlib import cm

class SPDN(EmbeddedManifold):
    """ manifold of symmetric positive definite matrices """

    def __init__(self,N=3):
        self.N = N
        dim = N*(N+1)//2
        emb_dim = N*N
        EmbeddedManifold.__init__(self,dim=dim,emb_dim=emb_dim)

        self.act = lambda g,q: jnp.tensordot(g,jnp.tensordot(q.reshape((N,N)),g,(1,1)),(1,0)).flatten()
        self.acts = jax.vmap(self.act,(0,None))

    def __str__(self):
        return "SPDN(%d), dim %d" % (self.N,self.dim)


    def plot(self, rotate=None, alpha = None):
        ax = plt.gca()
        #ax.set_aspect("equal")
        if rotate != None:
            ax.view_init(rotate[0],rotate[1])
    #     else:
    #         ax.view_init(35,225)
        plt.xlabel('x')
        plt.ylabel('y')


    def plot_path(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        assert(len(x.shape)>1)
        for i in range(x.shape[0]):
            self.plotx(x[i],
                  linewidth=linewidth if i==0 or i==x.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                  prevx=x[i-1] if i>0 else None,ellipsoid=ellipsoid,i=i,maxi=x.shape[0])
        return

    def plotx(self, x,color_intensity=1.,color=None,linewidth=3.,prevx=None,ellipsoid=None,i=None,maxi=None):
        x = x.reshape((self.N,self.N))
        (w,V) = np.linalg.eigh(x)
        s = np.sqrt(w[np.newaxis,:])*V # scaled eigenvectors
        if prevx is not None:
            prevx = prevx.reshape((self.N,self.N))
            (prevw,prevV) = np.linalg.eigh(prevx)
            prevs = np.sqrt(prevw[np.newaxis,:])*prevV # scaled eigenvectors
            ss = np.stack((prevs,s))

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        if ellipsoid is None:
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
                if prevx is not None:
                    plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=.3,color=colors[i])
        else:
            try:
                if i % int(ellipsoid['step']) != 0 and i != maxi-1:
                    return
            except:
                pass
            try:
                if ellipsoid['subplot']:
                    (fig,ax) = newfig3d(1,maxi//int(ellipsoid['step'])+1,i//int(ellipsoid['step'])+1,new_figure=i==0)
            except:
                (fig,ax) = newfig3d()
            #draw ellipsoid, from https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
            U, ss, rotation = np.linalg.svd(x)
            radii = np.sqrt(ss)
            u = np.linspace(0., 2.*np.pi, 20)
            v = np.linspace(0., np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for l in range(x.shape[0]):
                for k in range(x.shape[1]):
                    [x[l,k],y[l,k],z[l,k]] = np.dot([x[l,k],y[l,k],z[l,k]], rotation)
            ax.plot_surface(x, y, z, facecolors=cm.winter(y/np.amax(y)), linewidth=0, alpha=ellipsoid['alpha'])
            for i in range(s.shape[1]):
                plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1)
            plt.axis('off')
