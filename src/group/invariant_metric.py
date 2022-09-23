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
from src.utils import *

def initialize(G):
    """ add left-/right-invariant metric related structures to group """

    G.sigma = jnp.eye(G.dim,G.dim) # square root cometric / diffusion field
    G.sqrtA = G.inv(G.sigma) # square root metric
    G.A = jnp.tensordot(G.sqrtA,G.sqrtA,(0,0)) # metric
    G.W = G.inv(G.A) # covariance (cometric)
    def gV(v=None,w=None):
        if v is None and w is None:
            return G.A
        elif v is not None and w is None:
            return jnp.tensordot(G.A,v,(1,0))
        elif v.ndim == 1 and w.ndim == 1:
            return jnp.dot(v,jnp.dot(G.A,w))
        elif v.ndim == 1 and not w:
            return jnp.dot(G.A,v)
        elif v.ndim == 2 and w.ndim == 2:
            return jnp.tensordot(v,jnp.tensordot(G.A,w,(1,0)),(0,0))
        else:
            assert(False)
    G.gV = gV
    def cogV(cov=None,cow=None):
        if cov is None and cow is None:
            return G.W
        elif cov is not None and cow is None:
            return jnp.tensordot(G.W,cov,(1,0))
        elif cov.ndim == 1 and cow.ndim == 1:
            return jnp.dot(cov,jnp.dot(G.W,cow))
        elif cov.ndim == 2 and cow.ndim == 2:
            return jnp.tensordot(cov,jnp.tensordot(G.W,cow,(1,0)),(0,0))
        else:
            assert(False)
    G.cogV = cogV
    def gLA(xiv,xiw):
        v = G.LAtoV(xiv)
        w = G.LAtoV(xiw)
        return G.gV(v,w)
    G.gLA = gLA
    def cogLA(coxiv,coxiw):
        cov = G.LAtoV(coxiv)
        cow = G.LAtoV(coxiw)
        return G.cogV(cov,cow)
    G.cogLA = cogLA
    def gG(g,vg,wg):
        xiv = G.invpb(g,vg)
        xiw = G.invpb(g,wg)
        return G.gLA(xiv,xiw)
    G.gG = gG
    def gpsi(hatxi,v=None,w=None):
        g = G.psi(hatxi)
        vg = G.dpsi(hatxi,v)
        wg = G.dpsi(hatxi,w)
        return G.gG(g,vg,wg)
    G.gpsi = gpsi
    def cogpsi(hatxi,p=None,pp=None):
        invgpsi = G.inv(G.gpsi(hatxi))
        if p is not None and pp is not None:
            return jnp.tensordot(p,jnp.tensordot(invgpsi,pp,(1,0)),(0,0))
        elif p and not pp:
            return jnp.tensordot(invgpsi,p,(1,0))
        return invgpsi
    G.cogpsi = cogpsi

    # sharp/flat mappings
    def sharpV(mu):
        return jnp.dot(G.W,mu)
    G.sharpV = sharpV
    def flatV(v):
        return jnp.dot(G.A,v)
    G.flatV = flatV
    def sharp(g,pg):
        return G.invpf(g,G.VtoLA(jnp.dot(G.W,G.LAtoV(G.invcopb(g,pg)))))
    G.sharp = sharp
    def flat(g,vg):
        return G.invcopf(g,G.VtoLA(jnp.dot(G.A,G.LAtoV(G.invpb(g,vg)))))
    G.flat = flat
    def sharppsi(hatxi,p):
        return jnp.tensordot(G.cogpsi(hatxi),p,(1,0))
    G.sharppsi = sharppsi
    def flatpsi(hatxi,v):
        return jnp.tensordot(G.gpsi(hatxi),v,(1,0))
    G.flatpsi = flatpsi

