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
from src.utils import *

###############################################################
# Most probable paths for landmarks via development           #
###############################################################
def initialize(M,sigmas,a):
    """ Most probable paths for Kunita flows                 """
    """ M: shape manifold, a: flow field                     """
    
    def ode_MPP_landmarks(c,y):
        t,xlambd,chart = c
        qp, = y
        x = xlambd[0].reshape((M.N,M.m))  # points
        lambd = xlambd[1].reshape((M.N,M.m))
        
        sigmasx = sigmas(x)
        c = jnp.einsum('ri,rai->a',lambd,sigmasx)

        dx = a(x,qp)+jnp.einsum('a,rak->rk',c,sigmasx)
        dlambd = -jnp.einsum('ri,a,rairk->rk',lambd,c,jacrev(sigmas)(x))-jnp.einsum('ri,rirk->rk',lambd,jacrev(a)(x,qp))
        return jnp.stack((dx.flatten(),dlambd.flatten()))

    def chart_update_MPP_landmarks(xlambd,chart,y):
        if M.do_chart_update is None:
            return (xlambd,chart)
    
        lambd = xlambd[1].reshape((M.N,M.m))
        x = (xlambd[0],chart)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                                jnp.stack((new_x,M.update_covector(x,new_x,new_chart,lambd))),
                                xlambd),
                jnp.where(update,
                                new_chart,
                                chart))
    
    def MPP_landmarks(x,lambd,qps,dts):
        (ts,xlambds,charts) = integrate(ode_MPP_landmarks,chart_update_MPP_landmarks,jnp.stack((x[0],lambd)),x[1],dts,qps)
        return (ts,xlambds[:,0],xlambds[:,1],charts)
    M.MPP_landmarks = MPP_landmarks

