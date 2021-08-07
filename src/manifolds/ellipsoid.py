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

class Ellipsoid(EmbeddedManifold):
    """ 2d Ellipsoid """

    def chart(self):
        """ return default coordinate chart """
        if self.chart_center == 'x':
            return jnp.eye(3)[:,0]
        elif self.chart_center == 'y':
            return jnp.eye(3)[:,1]
        elif self.chart_center == 'z':
            return jnp.eye(3)[:,2]
        else:
            assert(False)

    def centered_chart(self,x):
        """ return centered coordinate chart """
        return x/self.params

    def get_B(self,v):
        """ R^3 basis with first basis vector v """
        b1 = v
        k = jnp.argmin(jnp.abs(v))
        ek = jnp.eye(3)[:,k]
        b2 = ek-v[k]*v
        b3 = cross(b1,b2)
        return jnp.stack((b1,b2,b3),axis=1)

    # Logarithm with standard Riemannian metric on S^2
    def StdLog(self, x,y):
        y = y/self.params # from ellipsoid to S^2
        proj = lambda x,y: jnp.dot(x,y)*x
        Fx = self.F(x)/self.params
        v = y-proj(Fx,y)
        theta = jnp.arccos(jnp.dot(Fx,y))
        normv = jnp.linalg.norm(v,2)
        w = theta/normv*v if normv >= 1e-5 else jnp.zeros_like(v)
        return jnp.dot(self.invJF((Fx,x[1])),self.params*w)

    def __init__(self,params=np.array([1.,1.,1.]),chart_center='z',use_spherical_coords=False):
        self.params = jnp.array(params) # ellipsoid parameters (e.g. [1.,1.,1.] for sphere)
        self.use_spherical_coords = use_spherical_coords
        self.chart_center = chart_center

        if not use_spherical_coords:
            F = lambda x: self.params*jnp.dot(self.get_B(x[1]),jnp.stack([-(-1+x[0][0]**2+x[0][1]**2),2*x[0][0],2*x[0][1]])/(1+x[0][0]**2+x[0][1]**2))
            def invF(x):
                Rinvx = jnp.linalg.solve(self.get_B(x[1]),x[0]/self.params)
                return jnp.stack([Rinvx[1]/(1+Rinvx[0]),Rinvx[2]/(1+Rinvx[0])])
            self.do_chart_update = lambda x: jnp.linalg.norm(x[0]) > .1 # look for a new chart if true
        # spherical coordinates, no charts
        self.F_spherical = lambda phitheta: self.params*jnp.stack([jnp.sin(phitheta[1]-np.pi/2)*jnp.cos(phitheta[0]),jnp.sin(phitheta[1]-np.pi/2)*jnp.sin(phitheta[0]),jnp.cos(phitheta[1]-np.pi/2)])
        self.JF_spherical = lambda x: jnp.jacobian(self.F_spherical(x),x)
        self.F_spherical_inv = lambda x: jnp.stack([jnp.arctan2(x[1],x[0]),jnp.arccos(x[2])])
        self.g_spherical = lambda x: jnp.dot(self.JF_spherical(x).T,self.JF_spherical(x))
        self.mu_Q_spherical = lambda x: 1./jnp.nlinalg.Det()(self.g_spherical(x))

        ## optionally use spherical coordinates in chart computations
        #if use_spherical_coords:
        #    F = lambda x: jnp.dot(x[1],self.F_spherical(x[0]))

        EmbeddedManifold.__init__(self,F,2,3,invF=invF)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))


    def __str__(self):
        return "%dd ellipsoid, parameters %s, spherical coords %s" % (self.dim,self.params,self.use_spherical_coords)

    def newfig(self):
        newfig3d()

    def plot(self,rotate=None,alpha=None,lw=0.3):
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
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=self.params[0]*np.cos(u)*np.sin(v)
        y=self.params[1]*np.sin(u)*np.sin(v)
        z=self.params[2]*np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.5)
    
        if alpha is not None:
            ax.plot_surface(x, y, z, color=cm.jet(0.), alpha=alpha)


    def plot_field(self, field,lw=.3):
        ax = plt.gca(projection='3d')
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

#        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #draw ellipsoid
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x=self.params[0]*np.cos(u)*np.sin(v)
        y=self.params[1]*np.sin(u)*np.sin(v)
        z=self.params[2]*np.cos(v)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Fx = np.array([x[i,j],y[i,j],z[i,j]])
                chart = self.centered_chart(Fx)
                xcoord = self.invF((Fx,chart))
                v = field((xcoord,chart))
                self.plotx((xcoord,chart),v=v)
