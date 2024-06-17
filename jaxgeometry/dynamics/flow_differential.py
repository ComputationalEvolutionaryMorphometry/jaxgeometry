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

###############################################################
# Compute differential d\phi along a phase-space path qt      #
# See Younes, Shapes and Diffeomorphisms, 2010 and            #
# Sommer et al., SIIMS 2013                                   #
###############################################################
def initialize(M):
    """ M: landmark manifold, scalar kernel                 """
    
    def ode_differential(c,y):
        t,dphi,chart = c
        qp, = y
        q = qp[0].reshape((M.N,M.m))  # points
        p = qp[1].reshape((M.N,M.m))  # points

        dk = M.dk_q(q,q)
        ddphi = jnp.einsum('iab,jic,jb->iac',dphi,dk,p)

        return ddphi 

    def chart_update_differential(dphi,chart,y):
        if M.do_chart_update is None:
            return (dphi,chart)
    
        assert(False) # not implemented yet
    
    def flow_differential(qps,dts):
        """ Transport covector lambd along covector path qps """
        (ts,dphis,charts) = integrate(ode_differential,chart_update_differential,jnp.tile(jnp.eye(M.m),(M.N,1,1)),None,dts,qps)
        return (ts,dphis,charts)
    M.flow_differential = flow_differential
