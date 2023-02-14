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

from src.plotting import *
import matplotlib.pyplot as plt

class Euclidean(Manifold):
    """ Euclidean space """

    def __init__(self,N=3):
        Manifold.__init__(self)
        self.dim = N

        self.update_coords = lambda coords,_: coords

        ##### Metric:
        self.g = lambda x: jnp.eye(self.dim)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))

    def __str__(self):
        return "Euclidean manifold of dimension %d" % (self.dim)

    def newfig(self):
        if self.dim == 2:
            newfig2d()
        elif self.dim == 3:
            newfig3d()

    def plot(self):
        if self.dim == 2:
            plt.axis('equal')
    
    def plot_path(self, xs, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, s=20, arrowcolor='k'):
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==N-1)
            prevx = x
        return

    def plotx(self, x, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, s=20, arrowcolor='k'):
        assert(type(x) == type(()) or x.shape[0] == self.dim)
        if type(x) == type(()):
            x = x[0]
        if type(prevx) == type(()):
            prevx = prevx[0]

        ax = plt.gca()

        if last:
            if self.dim == 2:
                plt.scatter(x[0],x[1],color=color,s=s)
            elif self.dim == 3:
                ax.scatter(x[0],x[1],x[2],color=color,s=s)
        else:
            try:
                xx = np.stack((prevx,x))
                if self.dim == 2:
                    plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
                elif self.dim == 3:
                    ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
            except:
                if self.dim == 2:
                    plt.scatter(x[0],x[1],color=color,s=s)
                elif self.dim == 3:
                    ax.scatter(x[0],x[1],x[2],color=color,s=s)

        try:
            plt.quiver(x[0], x[1], u[0], u[1], pivot='tail', linewidth=linewidth, scale=5, color=arrowcolor)
        except:
            pass

    
