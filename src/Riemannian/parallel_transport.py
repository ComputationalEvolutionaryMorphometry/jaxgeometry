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
    """ Riemannian parallel transport """

    def ode_parallel_transport(c,y):
        t,xv,prevchart = c
        x,chart,dx = y
        prevx = xv[0]
        v = xv[1]

        if M.do_chart_update is not None:
            dx = jnp.where(jnp.sum(jnp.square(chart-prevchart)) <= 1e-5,
                    dx,
                    M.update_vector((x,chart),prevx,prevchart,dx)
                )
        dv = -jnp.einsum('ikl,k,l->i',M.Gamma_g((x,chart)),dx,v)
        return jnp.stack((jnp.zeros_like(x),dv))
    
    def chart_update_parallel_transport(xv,prevchart,y):
        x,chart,dx = y
        if M.do_chart_update is None:
            return (xv,chart)

        prevx = xv[0]
        v = xv[1]
        return (jnp.where(jnp.sum(jnp.square(chart-prevchart)) <= 1e-5,
                                       jnp.stack((x,v)),
                                       jnp.stack((x,M.update_vector((prevx,prevchart),x,chart,v)))),
                chart)

    parallel_transport = lambda v,dts,xs,charts,dxs: integrate(ode_parallel_transport,chart_update_parallel_transport,jnp.stack((xs[0],v)),charts[0],dts,xs,charts,dxs)
    M.parallel_transport = jit(lambda v,dts,xs,charts,dxs: parallel_transport(v,dts,xs,charts,dxs)[1][:,1])
