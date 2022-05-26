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

def initialize(M,truncate_high_order_derivatives=False):
    """ add SR structure to manifold """
    """ currently assumes distribution and that ambient Riemannian manifold is Euclidean """

    d = M.dim

    if hasattr(M, 'D'):
        M.a = lambda x: jnp.dot(M.D(x),M.D(x).T)
    else:
        raise ValueError('no metric or cometric defined on manifold')

    ##### sharp map:
    M.sharp = lambda x,p: jnp.tensordot(M.a(x),p,(1,0))

    ##### Hamiltonian
    M.H = lambda x,p: .5*jnp.sum(jnp.dot(p,M.sharp(x,p))**2)

