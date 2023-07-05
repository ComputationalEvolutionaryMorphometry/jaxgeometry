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

###############################################################
# Eulerian / stochastic EPDiff acting on landmarks
###############################################################
def initialize(M,k=None):
    dq = jit(grad(M.H,argnums=1))
    dp = jit(lambda q,p: -gradx(M.H)(q,p))
    
    # noise basis
    if k is None: # use landmark kernel per default
        k = M.k
    
    k_q = lambda q1,q2: k(q1.reshape((-1,M.m))[:,np.newaxis,:]-q2.reshape((-1,M.m))[np.newaxis,:,:])
    K = lambda q1,q2: (k_q(q1,q2)[:,:,np.newaxis,np.newaxis]*jnp.eye(M.m)[np.newaxis,np.newaxis,:,:]).transpose((0,2,1,3)).reshape((M.dim,-1))

    def sde_Eulerian(c,y):
        t,x,chart,sigmas_x,sigmas_a = c
        dt,dW = y
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        
        sigmas_adW = sigmas_a*dW[:,np.newaxis]
        sigmadWq = jnp.tensordot(K(x[0],sigmas_x),sigmas_adW.flatten(),(1,0))
        sigmadWp = jnp.tensordot(
             jax.jacrev(
                 lambda lq: jnp.tensordot(K(lq,sigmas_x),sigmas_adW.flatten(),(1,0)).flatten(),
                 )(x[0]),
            x[1],(1,0))
    
        X = None # to be implemented
        det = jnp.stack((dqt,dpt))
        sto = jnp.stack((sigmadWq,sigmadWp))
        return (det,sto,X,jnp.zeros_like(sigmas_x),jnp.zeros_like(sigmas_a))

    def chart_update_Eulerian(xp,chart,*cy):
        if M.do_chart_update is None:
            return (xp,chart,*cy)
    
        p = xp[1]
        x = (xp[0],chart)
    
        update = M.do_chart_update(x)
        new_chart = M.centered_chart(M.F(x))
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                            jnp.stack((new_x,M.update_covector(x,new_x,new_chart,p))),
                            xp),
                jnp.where(update,
                            new_chart,
                            chart),
                *cy)

    M.Eulerian_qp = lambda q,p,sigmas_x,sigmas_a,dts,dWs: integrate_sde(sde_Eulerian,integrator_stratonovich,chart_update_Eulerian,jnp.stack((q[0],p)),q[1],dts,dWs,sigmas_x,sigmas_a)
    M.Eulerian = lambda q,p,sigmas_x,sigmas_a,dts,dWs: M.Eulerian_qp(q,p,sigmas_x,sigmas_a,dts,dWs)[0:3]
