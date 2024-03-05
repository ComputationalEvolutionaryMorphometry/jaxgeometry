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

def initialize(M):
    """ add sR structure to manifold """
    """ currently assumes distribution and that ambient Riemannian manifold is Euclidean """

    d = M.dim

    if not hasattr(M, 'D'):
        raise ValueError('no distribution defined on manifold')
        
    M.sR_dim = M.D(M.coords(jnp.zeros(M.dim))).shape[1]
    
    if not hasattr(M,'a'):
        M.a = lambda x: mmT(M.D(x))
    else:
        print('using existing M.a')
    
    ### trivial embedding
    M.F = lambda x: x[0]
    M.invF = lambda x: (x,M.chart())
    M.JF = jacfwdx(M.F)
    M.invJF = jacfwdx(M.invF)

    ##### sharp map:
    M.sharp = lambda x,p: jnp.tensordot(M.a(x),p,(1,0))

    ##### Hamiltonian
    if not hasattr(M,'H'):
        M.H = lambda x,p: 5*jnp.sum(jnp.einsum('i,ij->j',p,M.D(x))**2)
    else:
        print('using existing M.H')

    ##### divergence in divergence free othornormal distribution
    M.div = lambda x,X: jnp.einsum('ij,ji->',jacfwdx(X)(x)[:M.sR_dim,:],M.D(x))
