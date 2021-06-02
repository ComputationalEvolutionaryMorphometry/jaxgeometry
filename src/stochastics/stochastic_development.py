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
    """ development and stochastic development from R^d to M """

    # Deterministic development
    def ode_development(c,y):
        t,u,chart = c
        dgamma, = y

        u = (u,chart)
        nu = u[0][M.dim:].reshape((M.dim,-1))
        m = nu.shape[1]

        det = jnp.tensordot(M.Horizontal(u)[:,0:m], dgamma, axes = [1,0])
    
        return det

    M.development = jit(lambda u,dgamma,dts: integrate(ode_development,M.chart_update_FM,dts,u[0],u[1],dgamma))

    # Stochastic development
    def sde_development(c,y):
        t,u,chart = c
        dsm = y

        u = (u,chart)
        nu = u[0][M.dim:].reshape((M.dim,-1))
        m = nu.shape[1]

        sto = jnp.tensordot(M.Horizontal(u)[:,0:m], dsm, axes = [1,0])
    
        return (jnp.zeros_like(sto), sto, M.Horizontal(u)[:,0:m])

    M.sde_development = sde_development
    M.stochastic_development = jit(lambda u,dWt: integrate_sde(sde_development,integrator_stratonovich,M.chart_update_FM,u[0],u[1],dWt))
