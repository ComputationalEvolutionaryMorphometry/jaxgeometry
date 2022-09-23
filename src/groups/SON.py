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


from src.setup import *
from src.params import *

from src.groups.group import *

import matplotlib.pyplot as plt

class SON(LieGroup):
    """ Special Orthogonal Group SO(N) """

    def __init__(self,N=3,invariance='left'):
        dim = N*(N-1)//2 # group dimension
        LieGroup.__init__(self,dim,N,invariance=invariance)

        self.injectivity_radius = 2*jnp.pi

        # project to group (here using QR factorization)
        def to_group(g):
            (q,r) = jnp.linalg.qr(g)
            return jnp.dot(q,jnp.diag(jnp.diag(r)))

        ## coordinate chart linking Lie algebra LA={A\in\RR^{NxN}|\trace{A}=0} and V=\RR^G_dim
        # derived from https://stackoverflow.com/questions/25326462/initializing-a-symmetric-theano-dmatrix-from-its-upper-triangle
        r = jnp.arange(N)
        tmp_mat = r[jnp.newaxis, :] + ((N * (N - 3)) // 2-(r * (r - 1)) // 2)[::-1,jnp.newaxis]
        triu_index_matrix = jnp.triu(tmp_mat+1)-jnp.diag(jnp.diagonal(tmp_mat+1))

        def VtoLA(hatxi): # from \RR^G_dim to LA
            if hatxi.ndim == 1:
                m = jnp.concatenate((jnp.zeros(1),hatxi))[triu_index_matrix]
                return m-m.T
            else: # matrix
                m = jnp.concatenate((jnp.zeros((1,hatxi.shape[1])),hatxi))[triu_index_matrix,:]
                return m-m.transpose((1,0,2))
        self.VtoLA = VtoLA
        self.LAtoV = lambda m: m[np.triu_indices(N, 1)]

        #import theano.tensor.slinalg
        #Expm = jnp.slinalg.Expm()
        def Expm(g): # hardcoded for skew symmetric matrices to allow higher-order gradients
            (w,V) = jnp.linalg.eigh(1.j*g)
            w = -1j*w
            expm = jnp.real(jnp.tensordot(V,jnp.tensordot(jnp.diag(jnp.exp(w)),jnp.conj(V.T),(1,0)),(1,0)))
            return expm
        self.Expm = Expm
        self.Logm = lambda g : linalg.Logm()(g)#to_group(g))

        super(SON,self).initialize()

    def __str__(self):
        return "SO(%d) (dimension %d)" % (self.N,self.dim)

    def newfig(self):
        newfig3d()

    ### plotting
    import matplotlib.pyplot as plt
    def plot_path(self,g,color_intensity=1.,color=None,linewidth=3.,alpha=1.,prevg=None):
        assert(len(g.shape)>2)
        for i in range(g.shape[0]):
            self.plotg(g[i],
                  linewidth=linewidth if i==0 or i==g.shape[0]-1 else .3,
                  color_intensity=color_intensity if i==0 or i==g.shape[0]-1 else .7,
                  alpha=alpha,
                  prevg=g[i-1] if i>0 else None)
        return 

    def plotg(self,g,color_intensity=1.,color=None,linewidth=3.,alpha=1.,prevg=None):
        # Grid Settings:
        import matplotlib.ticker as ticker 
        ax = plt.gca()
        x = jnp.arange(-10,10,1)
        ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_yaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_zaxis.set_major_locator(ticker.FixedLocator(x))
        ax.w_xaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_yaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.w_zaxis.set_pane_color((0.98, 0.98, 0.99, 1.0))
        ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_zlim(-1.,1.)
        #ax.set_aspect("equal")
   
        s0 = jnp.eye(3) # shape
        s = jnp.dot(g,s0) # rotated shape
        if prevg is not None:
            prevs = jnp.dot(prevg,s0)

        colors = color_intensity*np.array([[1,0,0],[0,1,0],[0,0,1]])
        for i in range(s.shape[1]):
            plt.quiver(0,0,0,s[0,i],s[1,i],s[2,i],pivot='tail',linewidth=linewidth,color=colors[i] if color is None else color,arrow_length_ratio=.15,length=1,alpha=alpha)
            if prevg is not None:
                ss = jnp.stack((prevs,s))
                ss = ss/jnp.linalg.norm(ss,axis=1)[:,jnp.newaxis,:]
                plt.plot(ss[:,0,i],ss[:,1,i],ss[:,2,i],linewidth=1,color=colors[i])

