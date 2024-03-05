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
    """ Euler-Poincare geodesic integration """

    assert(G.invariance == 'left')

    def ode_EP(c,y):
        t,mu,_ = c
        xi = G.invFl(mu)
        dmut = -G.coad(xi,mu)
        return dmut
    G.EP = lambda mu,_dts=None: integrate(ode_EP,None,mu,None,dts() if _dts is None else _dts)

    # reconstruction
    def ode_EPrec(c,y):
        t,g,_ = c
        mu, = y
        xi = G.invFl(mu)
        dgt = G.dL(g,G.e,G.VtoLA(xi))
        return dgt
    G.EPrec = lambda g,mus,_dts=None: integrate(ode_EPrec,None,g,None,dts() if _dts is None else _dts,mus)

    ### geodesics
    G.coExpEP = lambda g,mu: G.EPrec(g,G.EP(mu)[1])[1][-1]
    G.ExpEP = lambda g,v: G.coExpEP(g,G.flatV(v))
    G.ExpEPpsi = lambda q,v: G.ExpEP(G.psi(q),G.flatV(v))
    G.coExpEPt = lambda g,mu: G.EPrec(g,G.EP(mu)[1])
    G.ExpEPt = lambda g,v: G.coExpEPt(g,G.flatV(v))
    G.ExpEPpsit = lambda q,v: G.ExpEPt(G.psi(q),G.flatV(v))
    G.DcoExpEP = lambda g,mu: jax.jaxrev(G.coExpEP)(g,mu)
