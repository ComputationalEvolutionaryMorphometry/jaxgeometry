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

def initialize(M):
    """ sub-Riemannian Brownian motion """

    def sde_Brownian_sR(c,y):
        t,x,chart = c
        dt,dW = y

        D = M.D((x,chart))
        # D0 = \sum_{i=1}^m div_\mu(X_i) X_i) - not implemented yet
        det = jnp.zeros_like(x) # Y^k(x)=X_0^k(x)+(1/2)\sum_{i=1}^m \langle \nabla X_i^k(x),X_i(x)\rangle
        sto = jnp.tensordot(D,dW,(1,0))
        return (det,sto,D)
    
    def chart_update_Brownian_sR(x,chart,y):
        if M.do_chart_update is None:
            return (x,chart,*y)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart((x,chart))
        new_x = M.update_coords((x,chart),new_chart)[0]

        return (jnp.where(update,
                                new_x,
                                x),
                jnp.where(update,
                                new_chart,
                                chart))
    
    M.sde_Brownian_sR = sde_Brownian_sR
    M.chart_update_Brownian_sR = chart_update_Brownian_sR
    M.Brownian_sR = jit(lambda x,dts,dWs: integrate_sde(sde_Brownian_sR,integrator_ito,chart_update_Brownian_sR,x[0],x[1],dts,dWs))
