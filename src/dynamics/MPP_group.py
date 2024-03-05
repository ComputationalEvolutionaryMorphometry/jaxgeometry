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

from src.group.quotient import *

###############################################################
# Most probable paths for Lie groups via development          #
###############################################################
def initialize(G,Sigma=None,a=None):
    """ Most probable paths and development """

    sign = -1. if G.invariance == 'right' else 1.
    Sigma = Sigma if Sigma is not None else jnp.eye(G.dim)

    def ode_mpp(sigma,c,y):
        t,alpha,_ = c
        
        at = a(t) if a is not None else jnp.zeros_like(alpha)
        
        z = jnp.dot(Sigma,G.sharpV(alpha))+at
        dalpha = sign*G.coad(z,alpha) # =-jnp.einsum('k,i,ijk->j',alpha,z,G.C) 
        return dalpha
    G.mpp = lambda alpha,dts,sigma=jnp.eye(G.dim): integrate(partial(ode_mpp,sigma),None,alpha,None,dts)

    # reconstruction
    def ode_mpprec(sigma,c,y):
        t,g,_ = c
        
        alpha, = y
        at = a(t) if a is not None else jnp.zeros_like(alpha)
        
        z = jnp.dot(Sigma,G.sharpV(alpha))+at
        dgt = G.invpf(g,G.VtoLA(z))
        return dgt
    G.mpprec = lambda g,alpha,dts,sigma=jnp.eye(G.dim): integrate(partial(ode_mpprec,sigma),None,g,None,dts,alpha)

    # tracking point (not reduced to Lie algebra) to allow point-depending drift
    def ode_mpp_drift(sigma,c,y):
        t,x,_ = c
        alpha = x[0:G.dim]
        g = x[G.dim:].reshape((G.dim,G.dim))
        
        at = jnp.linalg.solve(g,a(t,g)) if a is not None else jnp.zeros_like(alpha)
        
        z = jnp.dot(Sigma,G.sharpV(alpha))+at
        dalpha = sign*G.coad(z,alpha) # =-jnp.einsum('k,i,ijk->j',alpha,z,G.C) 
        dgt = G.invpf(g,G.VtoLA(z))
        return jnp.hstack((dalpha,dgt.flatten()))
    G.mpp_drift = lambda alpha,g,dts,sigma=jnp.eye(G.dim): integrate(partial(ode_mpp_drift,sigma),None,jnp.hstack((alpha,g.flatten())),None,dts)

    def MPP_forwardt(g,alpha,sigma,T=T,n_steps=n_steps):
        _dts = dts(T=T,n_steps=n_steps)
        (ts,alphas) = G.mpp(alpha,_dts,sigma)
        (ts,gs) = G.mpprec(g,alphas,_dts,sigma)
        
        return(gs,alphas)
    G.MPP_forwardt = MPP_forwardt
    
    # optimization to satisfy end-point conditions
    def MPP_f(g,alpha,y,sigma):
        gs,alphas = G.MPP_forwardt(g,alpha,sigma)
        gT = gs[-1]
        return (1./G.emb_dim)*jnp.sum(jnp.square(gT-y))
    
    def MPP(g,y,sigma=jnp.eye(G.dim)):
        res = jax.scipy.optimize.minimize(lambda alpha: MPP_f(g,alpha,y,sigma),jnp.zeros(G.dim),method='BFGS')
        alpha = res.x
        
        return alpha
    G.MPP = MPP

    def MPP_drift_f(g,alpha,y,sigma,proj,M,_dts):
        _,_,_,_,horz = horz_vert_split(g,proj,jnp.eye(G.dim),G,M)
        (ts,alphags) = G.mpp_drift(jnp.dot(horz,alpha),g,_dts,sigma)
        gT = alphags[-1,G.dim:].reshape((G.dim,G.dim))
        return (1./M.emb_dim)*jnp.sum(jnp.square(proj(gT)-M.F(y)))
    
    def MPP_drift(g,y,proj,M,sigma=jnp.eye(G.dim)):
        _dts = dts()
        res = jax.scipy.optimize.minimize(lambda alpha: MPP_drift_f(g,alpha,y,sigma,proj,M,_dts),jnp.zeros(M.dim),method='BFGS')
        _,_,_,_,horz = horz_vert_split(g,proj,jnp.eye(G.dim),G,M)
        alpha = jnp.dot(horz,res.x)
        
        return alpha
    G.MPP_drift = MPP_drift
    

# # Most probable paths
# def initialize(G,horz,vert,a=None):
#     """ Most probable paths and development """

#     assert(G.invariance == 'right')

#     def ode_mpp(sigma,c,y):
#         t,x,_ = c
        
#         vt = x[0]
#         ct = x[1]
#         lambdt = x[2]
#         at = a(t) if a is not None else jnp.zeros_like(vt)
#         Sigma = G.W(sigma)
        
#         domegat = jnp.dot(Sigma,vt)-at
#         dvt = horz(G.coad(domegat,vt-lambdt+ct))
#         dct = -vert(G.coad(domegat,vt))
#         dlambdt = vert(G.coad(domegat,ct))
#         return jnp.stack((dvt,ct,dlambdt))
#     G.mpp = lambda v,c,lambd,dts,sigma=jnp.eye(G.dim): integrate(partial(ode_mpp,sigma),None,jnp.stack((v,c,lambd)),None,dts)

#     # reconstruction
#     def ode_mpprec(sigma,c,y):
#         t,g,_ = c
        
#         x, = y
#         vt = x[0]
#         ct = x[1]
#         lambdt = x[2] 
#         at = a(t) if a is not None else jnp.zeros_like(vt)
#         Sigma = G.W(sigma)
        
#         domegat = jnp.dot(Sigma,vt)-at
#         dgt = G.invpf(g,G.VtoLA(domegat))
#         return dgt
#     G.mpprec = lambda g,vclambds,dts,sigma=jnp.eye(G.dim): integrate(partial(ode_mpprec,sigma),None,g,None,dts,vclambds)

#     def MPP_forwardt(g,v,c,lambd,sigma,T=T,n_steps=n_steps):
#         _dts = dts(T=T,n_steps=n_steps)
#         (ts,xs) = G.mpp(v,c,lambd,_dts,sigma)
#         (ts,gs) = G.mpprec(g,xs,_dts,sigma)
        
#         vs = xs[0]
#         cs = xs[1]
#         lambds = xs[2]
#         return(gs,cs)
#     G.MPP_forwardt = MPP_forwardt
    
#     # optimization to satisfy end-point conditions
#     def MPP_f(g,x,y,sigma):
#         x = x.reshape((3,G.dim))
#         v = x[0]
#         c = x[1]
#         lambd = x[2]
#         gs,cs = G.MPP_forwardt(g,v,c,lambd,sigma)
#         gT = gs[-1]
#         cT = cs[-1]
#         return (1./G.emb_dim)*jnp.sum(jnp.square(gT-y))+(1./G.dim)*jnp.sum(jnp.square(cT))
    
#     def MPP(g,y,sigma=jnp.eye(G.dim)):
#         res = jax.scipy.optimize.minimize(lambda x: MPP_f(g,x,y,sigma),jnp.zeros((3,G.dim)).flatten(),method='BFGS')
#         x = res.x.reshape((3,G.dim))
#         v = x[0]
#         c = x[1]
#         lambd = x[2]
        
#         return (v,c,lambd)
#     G.MPP = MPP
    
# E = jnp.eye(G.dim)
# horz = lambda v: jnp.dot(E[:,:2],jnp.dot(E[:,:2].T,v))
# vert = lambda v: jnp.dot(E[:,2:],jnp.dot(E[:,2:].T,v))
# initialize(G,horz,vert)
