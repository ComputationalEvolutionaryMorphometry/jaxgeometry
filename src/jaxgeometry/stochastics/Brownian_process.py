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

from jaxgeometry.setup import *
from jaxgeometry.utils import *

def initialize(G):
    """ Brownian motion with respect to left/right invariant metric """

    assert(G.invariance == 'left')

    def sde_Brownian_process(c,y):
        t,g,_,sigma = c
        dt,dW = y

        X = jnp.tensordot(G.invpf(g,G.eiLA),sigma,(2,0))
        det = jnp.zeros_like(g)
        sto = jnp.tensordot(X,dW,(2,0))
        return (det,sto,X,0.)

    G.sde_Brownian_process = sde_Brownian_process
    G.Brownian_process = lambda g,dts,dWt,sigma=jnp.eye(G.dim): integrate_sde(G.sde_Brownian_process,integrator_stratonovich,None,g,None,dts,dWt,sigma)[0:3]

