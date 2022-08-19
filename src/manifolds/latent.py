# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker
from scipy.stats import norm

class Latent(EmbeddedManifold):
    """ Latent space manifold define from embedding function F:R^dim->R^emb_dim, f e.g. a neural network """

    def __init__(self,F,dim,emb_dim,invF=None):
        EmbeddedManifold.__init__(self,F,dim,emb_dim,invF)

        # metric matrix
        self.g = lambda x: T.dot(self.JF(x).T,self.JF(x))

    def newfig(self):
        if self.emb_dim.eval() == 3:
            newfig3d()
        elif self.dim.eval() == 2:
            newfig2d()

    def plot(self, rotate=None, alpha=None, lw=0.3):
        if self.emb_dim.eval() == 3:
            ax = plt.gca(projection='3d')
            x = np.arange(-10, 10, 1)
            ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
            ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
            ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
            ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
            ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
            ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
            ax.xaxis._axinfo["grid"]['linewidth'] = lw
            ax.yaxis._axinfo["grid"]['linewidth'] = lw
            ax.zaxis._axinfo["grid"]['linewidth'] = lw
            ax.set_xlim(-1., 1.)
            ax.set_ylim(-1., 1.)
            ax.set_zlim(-1., 1.)
            ax.set_aspect("equal")
            if rotate is not None:
                ax.view_init(rotate[0], rotate[1])
                #     else:
                #         ax.view_init(35,225)
            plt.xlabel('x')
            plt.ylabel('y')

            #    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            # draw surface
            X, Y = np.meshgrid(norm.ppf(np.linspace(0.05, 0.95, 20)), norm.ppf(np.linspace(0.05, 0.95, 20)))
            xy = np.vstack([X.ravel(), Y.ravel()]).T
            xyz = np.apply_along_axis(self.Ff, 1, xy)
            x = xyz[:, 0].reshape(X.shape);
            y = xyz[:, 1].reshape(X.shape);
            z = xyz[:, 2].reshape(X.shape)
            print(z.shape)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)

            if alpha is not None:
                ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)

    # plot x on ellipsoid. x can be either in coordinates or in R^3
    def plotx(self, x, u=None, v=None, N_vec=np.arange(0,n_steps.eval()), i0=0, color='b', color_intensity=1., linewidth=1., s=15., prevx=None, last=True):
        if len(x.shape)>1:
            for i in range(x.shape[0]):
                self.plotx(x[i], u=u if i == 0 else None, v=v[i] if v is not None else None,
                           N_vec=N_vec,i0=i,
                           color=color,
                           color_intensity=color_intensity if i==0 or i==x.shape[0]-1 else .7,
                           linewidth=linewidth,
                           s=s,
                           prevx=x[i-1] if i>0 else None,
                           last=i==(x.shape[0]-1))
            return

        if self.emb_dim.eval() == 3:
            xcoords = x
            if x.shape[0] < 3: # map to embedding space
                x = self.Ff(x)

            ax = plt.gca(projection='3d')
            if prevx is None or last:
                ax.scatter(x[0],x[1],x[2],color=color,s=s)
            if prevx is not None:
                if prevx.shape[0] < 3:
                    prevx = self.Ff(prevx)
                xx = np.stack((prevx,x))
                ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)

            if u is not None:
                JFx = self.JFf(xcoords)
                u = np.dot(JFx, u)
                ax.quiver(x[0], x[1], x[2], u[0], u[1], u[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                          color='black')

            if v is not None:
                #Seq = lambda m, n: [t*n//m + n//(2*m) for t in range(m)]
                #Seqv = np.hstack([0,Seq(N_vec,n_steps.get_value())])
                if i0 in N_vec:#Seqv:
                    JFx = self.JFf(xcoords)
                    v = np.dot(JFx, v)
                    ax.quiver(x[0], x[1], x[2], v[0], v[1], v[2],
                              pivot='tail',
                              arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                              color='black')
        elif self.dim.eval() == 2:
            if prevx is None or last:
                plt.scatter(x[0],x[1],color=color,s=s)
            if prevx is not None:
                xx = np.stack((prevx,x))
                plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
            if v is not None:
                if i0 in N_vec:#Seqv:
                    plt.quiver(x[0], x[1], v[0], v[1], pivot='tail', linewidth=linewidth, color='black',
                               angles='xy', scale_units='xy', scale=1)

