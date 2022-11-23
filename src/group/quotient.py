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

def horz_vert_split(x,proj,sigma,G,M):
    # compute kernel of proj derivative with respect to inv A metric
    rank = M.dim
    Xframe = jnp.tensordot(G.invpf(x,G.eiLA),sigma,(2,0))
    Xframe_inv = jnp.linalg.pinv(Xframe.reshape((-1,G.dim)))
    dproj = jnp.einsum('...ij,ijk->...k',jacrev(proj)(x), Xframe)
    (_,_,Vh) = jnp.linalg.svd(jax.lax.stop_gradient(dproj),full_matrices=True)
    ns = Vh[rank:].T # null space
    proj_ns = jnp.tensordot(ns,ns,(1,1))    
    horz = Vh[0:rank].T # horz space
    proj_horz = jnp.tensordot(horz,horz,(1,1))
    
    return (Xframe,Xframe_inv,proj_horz,proj_ns,horz)

# hit target v at time t=Tend
def get_sde_fiber(sde_f,proj,G,M):
    def sde_fiber(c,y):
        (det,sto,X,*dys_sde) = sde_f(c,y)
        t,g,_,sigma = c
        dt,dW = y
        
        (Xframe,Xframe_inv,_,proj_ns,_) = horz_vert_split(g,proj,sigma,G,M)
        
        det = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(proj_ns,jnp.tensordot(Xframe_inv,X.reshape((-1,G.dim)),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_fiber

def get_sde_horz(sde_f,proj,G,M):
    def sde_horz(c,y):
        (det,sto,X,*dys_sde) = sde_f(c,y)
        t,g,_,sigma = c
        dt,dW = y
        
        (Xframe,Xframe_inv,proj_horz,_,_) = horz_vert_split(g,proj,sigma,G,M)        
        det = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,det.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,sto.flatten(),(1,0)),(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(proj_horz,jnp.tensordot(Xframe_inv,X.reshape((-1,G.dim)),(1,0)),(1,0)),(2,0)).reshape(X.shape)
        
        return (det,sto,X,*dys_sde)

    return sde_horz

def get_sde_lifted(sde_f,proj,G,M):
    def sde_lifted(c,y):
        t,g,chart,sigma,*cs = c
        dt,dW = y

        (det,sto,X,*dys_sde) = sde_f((t,M.invF((proj(g),chart)),chart,*cs),y)
        
        (Xframe,Xframe_inv,proj_horz,_,horz) = horz_vert_split(g,proj,sigma,G,M) 

        
        det = jnp.tensordot(Xframe,jnp.tensordot(horz,det,(1,0)),(2,0)).reshape(g.shape)
        sto = jnp.tensordot(Xframe,jnp.tensordot(horz,sto,(1,0)),(2,0)).reshape(g.shape)
        X = jnp.tensordot(Xframe,jnp.tensordot(horz,X,(1,0)),(2,0)).reshape((G.dim,G.dim,M.dim))
        
        return (det,sto,X,jnp.zeros_like(sigma),*dys_sde)

    return sde_lifted

## find g in fiber above x closests to g0
#from scipy.optimize import minimize
#def lift_to_fiber(x,x0,G,M):
#    shoot = lambda hatxi: G.gV(hatxi,hatxi)
#    try:
#        hatxi = minimize(shoot,
#                np.zeros(G.dim),
#                method='COBYLA',
#                constraints={'type':'ineq','fun':lambda hatxi: np.min((G.injectivity_radius-np.max(hatxi),
#                                                                      1e-8-np.linalg.norm(M.act(G.exp(G.VtoLA(hatxi)),x0)-x)**2))},
#                ).x
#        hatxi = minimize(lambda hatxi: np.linalg.norm(M.act(G.exp(G.VtoLA(hatxi)),x0)-x)**2,
#                         hatxi).x # fine tune    
#    except AttributeError: # injectivity radius not defined
#        hatxi = minimize(shoot,
#                np.zeros(G.dim),
#                method='COBYLA',
#                constraints={'type':'ineq','fun':lambda hatxi: 1e-8-np.linalg.norm(M.act(G.exp(G.VtoLA(hatxi)),x0)-x)**2}).x
#        hatxi = minimize(lambda hatxi: np.linalg.norm(M.act(G.exp(G.VtoLA(hatxi)),x0)-x)**2,
#                         hatxi).x # fine tune
#    l0 = G.exp(G.VtoLA(hatxi))
#    try: # project to group if to_group function is available
#        l0 = G.to_group(l0)
#    except NameError:
#        pass
#    return (l0,hatxi)
#
## estimate fiber volume
#import scipy.special
#from src.plotting import *
#
#def fiber_samples(G,Brownian_fiberf,L,pars):
#    (seed,) = pars
#    if seed:
#        srng.seed(seed)
#    gsl = np.zeros((L,) + G.e.shape)
#    dsl = np.zeros(L)
#    (ts, gs) = Brownian_fiber(G.e, dWs(G.dim))
#    vl = gs[-1]  # starting point
#    for l in range(L):
#        (ts, gs) = Brownian_fiber(vl, dWs(G.dim))
#        gsl[l] = gs[-1]
#        dsl[l] = np.linalg.norm(G.LAtoV(G.log(gs[-1])))  # distance to sample with canonical biinvariant metric
#        vl = gs[-1]
#
#    return (gsl, dsl)
#
#def estimate_fiber_volume(G, M, lfiber_samples, nr_samples=100, plot_dist_histogram=False, plot_samples=False):
#    """ estimate fiber volume with restricted Riemannian G volume element (biinvariant metric) """
#    L = nr_samples // (cpu_count() // 2)  # samples per processor
#
#    try:
#        mpu.openPool()
#        sol = mpu.pool.imap(partial(lfiber_samples, L), mpu.inputArgs(np.random.randint(1000, size=cpu_count() // 2)))
#        res = list(sol)
#        gsl = mpu.getRes(res, 0).reshape((-1,) + G.e.shape)
#        dsl = mpu.getRes(res, 1).flatten()
#    except:
#        mpu.closePool()
#        raise
#    else:
#        mpu.closePool()
#
#    if plot_dist_histogram:
#        # distance histogram
#        plt.hist(dsl, 20)
#
#    if plot_samples:
#        # plot samples
#        newfig()
#        for l in range(0, L):
#            G.plotg(gsl[l])
#        plt.show()
#
#    # count percentage of samples below distance d to e relative to volume of d-ball
#    d = np.max(dsl)  # distance must be smaller than fiber radius
#    fiber_dim = G.dim - M.dim
#    ball_volume = np.pi ** (fiber_dim / 2) / scipy.special.gamma(fiber_dim / 2 + 1) * d ** fiber_dim
#
#    return ball_volume / (np.sum(dsl < d) / (dsl.size))
