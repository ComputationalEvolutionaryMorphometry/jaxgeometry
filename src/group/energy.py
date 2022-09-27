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

def initialize(G):
    """ group Lagrangian and Hamiltonian from invariant metric """

    # Lagrangian
    def Lagrangian(g,vg):
        return .5*G.gG(g,vg,vg)
    G.Lagrangian = Lagrangian
    # Lagrangian using psi map
    def Lagrangianpsi(q,v):
        return .5*G.gpsi(q,v,v)
    G.Lagrangianpsi = Lagrangianpsi
    G.dLagrangianpsidq = jax.grad(G.Lagrangianpsi)
    G.dLagrangianpsidv = jax.grad(G.Lagrangianpsi)
    # LA restricted Lagrangian
    def l(hatxi):
        return 0.5*G.gV(hatxi,hatxi)
    G.l = l
    G.dldhatxi = jax.grad(G.l)

    # Hamiltonian using psi map
    def Hpsi(q,p):
        return .5*G.cogpsi(q,p,p)
    G.Hpsi = Hpsi
    # LA^* restricted Hamiltonian
    def Hminus(mu):
        return .5*G.cogV(mu,mu)
    G.Hminus = Hminus
    G.dHminusdmu = jax.grad(G.Hminus)

    # Legendre transformation. The above Lagrangian is hyperregular
    G.FLpsi = lambda q,v: (q,G.dLagrangianpsidv(q,v))
    G.invFLpsi = lambda q,p: (q,G.cogpsi(q,p))
    def HL(q,p):
        (q,v) = invFLpsi(q,p)
        return jnp.dot(p,v)-L(q,v)
    G.HL = HL
    G.Fl = lambda hatxi: G.dldhatxi(hatxi)
    G.invFl = lambda mu: G.cogV(mu)
    def hl(mu):
        hatxi = invFl(mu)
        return jnp.dot(mu,hatxi)-l(hatxi)
    G.hl = hl

    # default Hamiltonian
    G.H = lambda q,p: G.Hpsi(q[0],p) if type(q) == type(()) else G.Hpsi(q,p)

# A.set_value(np.diag([3,2,1]))
# print(FLpsif(q0,v0))
# print(invFLpsif(q0,p0))
# (flq0,flv0)=FLpsif(q0,v0)
# print(q0,v0)
# print(invFLpsif(flq0,flv0))
