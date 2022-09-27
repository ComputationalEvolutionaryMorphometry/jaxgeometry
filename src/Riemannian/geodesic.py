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

def initialize(M):
    def ode_geodesic(c,y):
        t,x,chart = c
        dx2t = -jnp.einsum('ikl,k,l->i',M.Gamma_g((x[0],chart)),x[1],x[1])
        dx1t = x[1] 
        return jnp.stack((dx1t,dx2t))

    def chart_update_geodesic(xv,chart,y):
        if M.do_chart_update is None:
            return (xv,chart)
    
        v = xv[1]
        x = (xv[0],chart)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                                jnp.stack((new_x,M.update_vector(x,new_x,new_chart,v))),
                                xv),
                jnp.where(update,
                                new_chart,
                                chart))
    
    M.geodesic = jit(lambda x,v,dts: integrate(ode_geodesic,chart_update_geodesic,jnp.stack((x[0],v)),x[1],dts))
    
    def Exp(x,v,T=T,n_steps=n_steps):
        curve = M.geodesic(x,v,dts(T,n_steps))
        x = curve[1][-1,0]
        chart = curve[2][-1]
        return(x,chart)
    M.Exp = Exp
    def Expt(x,v,T=T,n_steps=n_steps):
        curve = M.geodesic(x,v,dts(T,n_steps))
        xs = curve[1][:,0]
        charts = curve[2]
        return(xs,charts)
    M.Expt = Expt
