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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class Cylinder(EmbeddedManifold):
    """ 2d Cylinder """

    def chart(self):
        """ return default coordinate chart """
        return jnp.zeros(self.dim)

    def centered_chart(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            Fx = jax.lax.stop_gradient(self.F(x))
        else:
            Fx = x # already in embedding space
        return self.invF((Fx,self.chart()))  # chart centered at coords

    def get_B(self,v):
        """ R^3 basis with first basis vector v """
        b1 = v
        k = jnp.argmin(jnp.abs(v))
        ek = jnp.eye(3)[:,k]
        b2 = ek-v[k]*v
        b3 = cross(b1,b2)
        return jnp.stack((b1,b2,b3),axis=1)

    # Logarithm with standard Riemannian metric
    def StdLog(self,_x,y): 
        (x,chart) = self.update_coords(_x,self.centered_chart(self.F(_x)))
        y = self.invF((y,chart))
        return self.update_vector((x,chart),_x[0],_x[1],y-x)
    
    def __init__(self,params=(1.,jnp.array([0.,1.,0.]),0.)):
        self.radius = params[0] # axis of cylinder
        self.orientation = jnp.array(params[1]) # axis of cylinder
        self.theta = params[2] # angle around rotation axis

        F = lambda x: jnp.dot(self.get_B(self.orientation),
                jnp.stack([x[0][1]+x[1][1],self.radius*jnp.cos(self.theta+x[1][0]+x[0][0]),self.radius*jnp.sin(self.theta+x[1][0]+x[0][0])]))
        def invF(x):
            Rinvx = jnp.linalg.solve(self.get_B(self.orientation),x[0])
            rotangle = -(self.theta+x[1][0])
            rot = jnp.dot(jnp.stack(
                (jnp.stack((jnp.cos(rotangle),-jnp.sin(rotangle))),
                 jnp.stack((jnp.sin(rotangle),jnp.cos(rotangle))))),
                Rinvx[1:])
            return jnp.stack([jnp.arctan2(rot[1],rot[0]),Rinvx[0]-x[1][1]])
        self.do_chart_update = lambda x: jnp.max(jnp.abs(x[0])) >= np.pi/4 # look for a new chart if true

        EmbeddedManifold.__init__(self,F,2,3,invF=invF)

    def __str__(self):
        return "cylinder in R^3, radius %s, axis %s, rotation around axis %s" % (self.radius,self.orientation,self.theta)

    def newfig(self):
        newfig3d()

    def plot(self, rotate=None,alpha=None,lw=0.3):
        ax = plt.gca()
        x = np.arange(-10,10,1)
        ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.xaxis._axinfo["grid"]['linewidth'] = lw
        ax.yaxis._axinfo["grid"]['linewidth'] = lw
        ax.zaxis._axinfo["grid"]['linewidth'] = lw
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_zlim(-1.,1.)
        #ax.set_aspect("equal")
        if rotate is not None:
            ax.view_init(rotate[0],rotate[1])
    #     else:
    #         ax.view_init(35,225)
        plt.xlabel('x')
        plt.ylabel('y')

    #    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #draw ellipsoid
        u, v = np.mgrid[-np.pi:np.pi:20j, -np.pi:np.pi:10j]
        x = np.zeros(u.shape)
        y = np.zeros(u.shape)
        z = np.zeros(u.shape)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                w = self.F(self.coords(jnp.array([u[i,j],v[i,j]])))
                x[i,j] = w[0]; y[i,j] = w[1]; z[i,j] = w[2]
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)

        if alpha is not None:
            ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)


    def plot_field(self, field,lw=.3):
        ax = plt.gca()
        x = np.arange(-10,10,1)
        ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.xaxis._axinfo["grid"]['linewidth'] = lw
        ax.yaxis._axinfo["grid"]['linewidth'] = lw
        ax.zaxis._axinfo["grid"]['linewidth'] = lw
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_zlim(-1.,1.)
        #ax.set_aspect("equal")

        plt.xlabel('x')
        plt.ylabel('y')

        u, v = np.mgrid[-np.pi:np.pi:40j, -np.pi:np.pi:20j]
        x = np.zeros(u.shape)
        y = np.zeros(u.shape)
        z = np.zeros(u.shape)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                w = self.F(self.coords(jnp.array([u[i,j],v[i,j]])))
                x[i,j] = w[0]; y[i,j] = w[1]; z[i,j] = w[2]
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Fx = np.array([x[i,j],y[i,j],z[i,j]])
                chart = self.centered_chartf(Fx)
                xcoord = self.invFf((Fx,chart))
                v = field((xcoord,chart))
                self.plotx((xcoord,chart),v=v)



