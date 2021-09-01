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
# Langevin equations https://arxiv.org/abs/1605.09276
###############################################################
def initialize(M):
    dq = jit(grad(M.H,argnums=1))
    dp = jit(lambda q,p: -gradx(M.H)(q,p))

    def sde_Langevin(c,y):
        t,x,chart,l,s = c
        dW, = y
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])-l*dq((x[0],chart),x[1])

        X = jnp.stack((jnp.zeros((M.dim,M.dim)),s*jnp.eye(M.dim)))
        det = jnp.stack((dqt,dpt))
        sto = jnp.tensordot(X,dW,(1,0))
        return (det,sto,X,jnp.zeros_like(l),jnp.zeros_like(s))

    def chart_update_Langevin(xp,chart,cy):
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

    M.Langevin_qp = lambda q,p,l,s,dWt: integrate_sde(sde_Langevin,integrator_ito,chart_update_Langevin,jnp.stack((q[0],p)),q[1],dWt,l,s)

    M.Langevin = lambda q,p,l,s,dWt: M.Langevin_qp(q,p,l,s,dWt)[0:3]
