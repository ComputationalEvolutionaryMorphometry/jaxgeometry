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

def initialize(G,Psi=None,r=None):
    """ stochastic coadjoint motion with left/right invariant metric
    see Noise and dissipation on coadjoint orbits arXiv:1601.02249 [math.DS]
    and EulerPoincare.py """

    assert(G.invariance == 'left')

    # Matrix function Psi:LA\rightarrow R^r must be defined beforehand
    # example here from arXiv:1601.02249
    if Psi is None:
        sigmaPsi = jnp.eye(G.dim)
        Psi = lambda mu: jnp.dot(sigmaPsi,mu)
        # r = Psi.shape[0]
        r = G.dim
    assert(Psi is not None and r is not None)

    def sde_stochastic_coadjoint(c,y):
        t,mu,_ = c
        dt,dW = y

        xi = G.invFl(mu)
        det = -G.coad(xi,mu)
        Sigma = G.coad(mu,jax.jacrev(Psi)(mu).transpose((1,0)))
        sto = jnp.tensordot(Sigma,dW,(1,0))
        return (det,sto,Sigma)
    G.sde_stochastic_coadjoint = sde_stochastic_coadjoint
    G.stochastic_coadjoint = lambda mu,dts,dWt: integrate_sde(G.sde_stochastic_coadjoint,integrator_stratonovich,None,mu,None,dts,dWt)

    # reconstruction as in Euler-Poincare / Lie-Poisson reconstruction
    if not hasattr(G,'EPrec'):
        from jaxgeometry.group import EulerPoincare
        EulerPoincare.initialize(G)
    G.stochastic_coadjointrec = G.EPrec

