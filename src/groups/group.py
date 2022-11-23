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

from src.manifolds.manifold import *

class LieGroup(EmbeddedManifold):
    """ Base Lie Group class """

    def __init__(self,dim,N,invariance='left'):
        EmbeddedManifold.__init__(self)

        self.dim = dim
        self.N = N # N in SO(N)
        self.emb_dim  = N*N # matrix/embedding space dimension
        self.invariance = invariance

        self.e = jnp.eye(N,N) # identity element
        self.zeroLA = jnp.zeros((N,N)) # zero element in LA
        self.zeroV = jnp.zeros((self.dim,)) # zero element in V

    def initialize(self):
        """ Initial group operations. To be called by sub-classes after definition of dimension, Expm etc.

        Notation:
            hatxi # \RR^G_dim vector
            xi # matrix in LA
            eta # matrix in LA
            alpha # matrix in LA^*
            beta # matrix in LA^*
            g # \RR^{NxN} matrix
            gs # sequence of \RR^{NxN} matrices
            h # \RR^{NxN} matrix
            vg # \RR^{NxN} tangent vector at g
            wg # \RR^{NxN} tangent vector at g
            vh # \RR^{NxN} tangent vector at h
            w # \RR^G_dim tangent vector in coordinates
            v # \RR^G_dim tangent vector in coordinates
            pg # \RR^{NxN} cotangent vector at g
            ph # \RR^{NxN} cotangent vector at h
            p # \RR^G_dim cotangent vector in coordinates
            pp # \RR^G_dim cotangent vector in coordinates
            mu # \RR^G_dim LA cotangent vector in coordinates
        """

        ## group operations
        self.inv = lambda g: jnp.linalg.inv(g)

        ## group exp/log maps
        self.exp = self.Expm
        def expt(xi,_dts=None):
            if _dts is None: _dts = dts()
            return lax.scan(lambda t,dt: (t+dt,self.exp(t*xi)),0.,_dts)
        self.expt = expt
        self.log = self.Logm

        ## Lie algebra
        self.eiV = jnp.eye(self.dim) # standard basis for V
        self.eiLA = self.VtoLA(self.eiV) # pushforward eiV basis for LA
        #stdLA = jnp.eye(N*N,N*N).reshape((N,N,N*N)) # standard basis for \RR^{NxN}
        #eijV = jnp.eye(G_dim) # standard basis for V
        #eijLA = jnp.zeros((N,N,G_dim)) # eij in LA
        def bracket(xi,eta):
            if xi.ndim == 2 and eta.ndim == 2:
                return jnp.tensordot(xi,eta,(1,0))-jnp.tensordot(eta,xi,(1,0))
            elif xi.ndim == 3 and eta.ndim == 3:
                return jnp.tensordot(xi,eta,(1,0)).dimshuffle((0,2,1,3))-jnp.tensordot(eta,xi,(1,0)).dimshuffle((0,2,1,3))
            else:
                assert(False)
        self.bracket =  bracket
        #C = bracket(eiLA,eiLA) # structure constants, debug
        #C = jnp.linalg.lstsq(eiLA.reshape((N*N*G_dim*G_dim,G_dim*G_dim*G_dim)),bracket(eiLA,eiLA).reshape((N*N*G_dim*G_dim))).reshape((G_dim,G_dim,G_dim)) # structure constants
        self.C = jnp.zeros((self.dim,self.dim,self.dim)) # structure constants
        for i in range(self.dim):
            for j in range(self.dim):
                xij = self.bracket(self.eiLA[:,:,i],self.eiLA[:,:,j])
                #lC[i,j,:] = jnp.linalg.lstsq(
                #    self.eiLA.reshape((self.N*self.N,self.dim)),
                #    xij.flatten(),
                #    rcond=-1
                #)[0]
                self.C = self.C.at[i,j].set(jnp.linalg.lstsq(
                            self.eiLA.reshape(self.N*self.N, self.dim),
                            xij.reshape(self.N*self.N)
                            )[0])

        ## surjective mapping \psi:\RR^G_dim\rightarrow G
        self.psi = lambda hatxi: self.exp(self.VtoLA(hatxi))
        self.invpsi = lambda g: self.LAtoV(self.log(g))
        def dpsi(hatxi,v=None):
            dpsi = jax.jacrev(self.psi)(hatxi)
            if v:
                return jnp.tensordot(dpsi,v,(2,0))
            return dpsi
        self.dpsi = dpsi
        def dinvpsi(g,vg=None):
            dinvpsi = jax.jacrev(self.invpsi)(g)
            if vg:
                return jnp.tensordot(dinvpsi,vg,((1,2),(0,1)))
            return dinvpsi
        self.dinvpsi = dinvpsi        

        ## left/right translation
        self.L = lambda g,h: jnp.tensordot(g,h,(1,0)) # left translation L_g(h)=gh
        self.R = lambda g,h: jnp.tensordot(h,g,(1,0)) # right translation R_g(h)=hg
        # pushforward of L/R of vh\in T_hG
        #dL = lambda g,h,vh: theano.gradient.Rop(L(theano.gradient.disconnected_grad(g),h).flatten(),h,vh).reshape((N,N))
        def dL(g,h,vh=None):
            dL = jax.jacrev(self.L,1)(g,h)
            if vh is not None:
                return jnp.tensordot(dL,vh,((2,3),(0,1)))
            return dL
        self.dL = dL
        def dR(g,h,vh=None):
            dR = jax.jacrev(self.R,1)(g,h)
            if vh is not None:
                return jnp.tensordot(dR,vh,((2,3),(0,1)))
            return dR
        self.dR = dR
        # pullback of L/R of vh\in T_h^*G
        self.codL = lambda g,h,vh: self.dL(g,h,vh).T
        self.codR = lambda g,h,vh: self.dR(g,h,vh).T

        ## actions
        self.Ad = lambda g,xi: self.dR(self.inv(g),g,self.dL(g,self.e,xi))
        self.ad = lambda xi,eta: self.bracket(xi,eta)
        self.coad = lambda v,p: jnp.tensordot(jnp.tensordot(self.C,v,(0,0)),p,(1,0))

        ## invariance
        if self.invariance == 'left':
            self.invtrns = self.L # invariance translation
            self.invpb = lambda g,vg: self.dL(self.inv(g),g,vg) # left invariance pullback from TgG to LA
            self.invpf = lambda g,xi: self.dL(g,self.e,xi) # left invariance pushforward from LA to TgG
            self.invcopb = lambda g,pg: self.codL(self.inv(g),g,pg) # left invariance pullback from Tg^*G to LA^*
            self.invcopf = lambda g,alpha: self.codL(g,self.e,alpha) # left invariance pushforward from LA^* to Tg^*G
            self.infgen = lambda xi,g: self.dR(g,self.e,xi) # infinitesimal generator
        else:
            self.invtrns = self.R # invariance translation
            self.invpb = lambda g,vg: self.dR(self.inv(g),g,vg) # right invariance pullback from TgG to LA
            self.invpf = lambda g,xi: self.dR(g,self.e,xi) # right invariance pushforward from LA to TgG
            self.invcopb = lambda g,pg: self.codR(self.inv(g),g,pg) # right invariance pullback from Tg^*G to LA^*
            self.invcopf = lambda g,alpha: self.codR(g,self.e,alpha) # right invariance pushforward from LA^* to Tg^*G
            self.infgen = lambda xi,g: self.dL(g,self.e,xi) # infinitesimal generator

    def __str__(self):
        return "abstract Lie group"

