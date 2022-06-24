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

def initialize(M,sde,chart_update,integrator=integrator_ito):
    """ product diffusions """

    def sde_product(c,y):
        t,x,chart,*cy = c
        dt,dW = y
        
        (det,sto,X,*dcy) = jax.vmap(lambda x,chart,dW: sde((t,x,chart,*cy),(dt,dW)),0)(x,chart,dW)

        return (det,sto,X,*dcy)
    
    # note: y is not parametrized on product and thus cannot be updated by chart update
    def chart_update_product(x,chart,*ys):
        if M.do_chart_update is None:
            return (x,chart,*ys)

        return (*jax.vmap(lambda x,chart: chart_update(x,chart,*ys),0)(x,chart),*ys)
    
    M.sde_product = sde_product
    M.chart_update_product = chart_update_product
    M.product = jit(lambda x,dts,dWs: integrate_sde(sde_product,integrator,chart_update_product,x[0],x[1],dts,dWs))

