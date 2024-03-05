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

def initialize(M,sde_product,chart_update_product,integrator=integrator_ito,T=1):
    """ diagonally conditioned product diffusions """

    def sde_diagonal(c,y):
        if M.do_chart_update is None:
            t,x,chart,T,*cy = c
        else:
            t,x,chart,T,ref_chart,*cy = c
        dt,dW = y
        
        (det,sto,X,*dcy) = sde_product((t,x,chart,*cy),y)

        if M.do_chart_update is None:
            xref = x
        else:
            xref = jax.vmap(lambda x,chart: M.update_coords((x,chart),ref_chart)[0],0)(x,chart)
        m = jnp.mean(xref,0) # mean
        href = jax.lax.cond(t<T-dt/2,
                 lambda _: (m-xref)/(T-t),
                 lambda _: jnp.zeros_like(det),
                 None)
        if M.do_chart_update is None:
            h = href
        else:
            h = jax.vmap(lambda xref,x,chart,h: M.update_vector((xref,ref_chart),x,chart,h),0)(xref,x,chart,href)
        
        # jnp.tensordot(X,h,(2,1))
        if M.do_chart_update is None:
            return (det+h,sto,X,0.,*dcy)
        else:
            return (det+h,sto,X,0.,jnp.zeros_like(ref_chart),*dcy)

    def chart_update_diagonal(x,chart,*ys):
        if M.do_chart_update is None:
            return (x,chart,*ys)

        (ref_chart,T,*_ys) = ys

        (new_x,new_chart,*new_ys) = chart_update_product(x,chart,*_ys)
        return (new_x,new_chart,ref_chart,T,*new_ys)
    
    M.sde_diagonal = sde_diagonal
    M.chart_update_diagonal = chart_update_product
    if M.do_chart_update is None:
        M.diagonal = jit(lambda x,dts,dWt: integrate_sde(sde_diagonal,integrator,M.chart_update_diagonal,x[0],x[1],dts,dWt,jnp.sum(dts))[0:3])
    else:
        M.diagonal = jit(lambda x,dts,dWt,ref_chart,*ys: integrate_sde(sde_diagonal,integrator,chart_update_diagonal,x[0],x[1],dts,dWt,jnp.sum(dts),ref_chart,*ys)[0:3])

