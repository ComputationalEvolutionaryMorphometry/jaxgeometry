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
    """ Lie-Poisson geodesic integration """

    assert(G.invariance == 'left')

    def ode_LP(c,y):
        t,mu,_ = c
        dmut = G.coad(G.dHminusdmu(mu),mu)
        return dmut
    G.LP = lambda mu,_dts=None: integrate(ode_LP,None,mu,None,dts() if _dts is None else _dts)

    # reconstruction
    def ode_LPrec(c,y):
        t,g,_ = c
        mu, = y
        dgt = G.dL(g,G.e,G.VtoLA(G.dHminusdmu(mu)))
        return dgt
    G.LPrec = lambda g,mus,_dts=None: integrate(ode_LPrec,None,g,None,dts() if _dts is None else _dts,mus)

    ### geodesics
    G.coExpLP = lambda g,mu: G.LPrec(g,G.LP(mu)[1])[1][-1]
    G.ExpLP = lambda g,v: G.coExpLP(g,G.flatV(v))
    G.coExpLPt = lambda g,mu: G.LPrec(g,G.LP(mu)[1])
    G.ExpLPt = lambda g,v: G.coExpLPt(g,G.flatV(v))
    G.DcoExpLP = lambda g,mu: jax.jacrev(G.coExp)(g,mu)
