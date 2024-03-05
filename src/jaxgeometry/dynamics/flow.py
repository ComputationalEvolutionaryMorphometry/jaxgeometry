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
    """ flow along a vector field X """
    def flow(X):

        def ode_flow(c,y):
            t,x,chart = c
            return X((x,chart))
        
        def chart_update_flow(x,chart,*ys):
            if M.do_chart_update is None:
                return (x,chart)

            update = M.do_chart_update(x)
            new_chart = M.centered_chart((x,chart))
            new_x = M.update_coords((x,chart),new_chart)[0]

            return (jnp.where(update,
                                    new_x,
                                    x),
                    jnp.where(update,
                                    new_chart,
                                    chart),
                    )
        
        flow = jit(lambda x,dts: integrate(ode_flow,chart_update_flow,x[0],x[1],dts))
        return flow
    M.flow = flow
