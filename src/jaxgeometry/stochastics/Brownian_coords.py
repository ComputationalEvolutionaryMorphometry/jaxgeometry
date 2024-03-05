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
    """ Brownian motion in coordinates """

    def sde_Brownian_coords(c,y):
        t,x,chart,s = c
        dt,dW = y

        gsharpx = M.gsharp((x,chart))
        X = s*jnp.linalg.cholesky(gsharpx)
        det = -.5*(s**2)*jnp.einsum('kl,ikl->i',gsharpx,M.Gamma_g((x,chart)))
        sto = jnp.tensordot(X,dW,(1,0))
        return (det,sto,X,0.)
    
    def chart_update_Brownian_coords(x,chart,*ys):
        if M.do_chart_update is None:
            return (x,chart,*ys)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart((x,chart))
        new_x = M.update_coords((x,chart),new_chart)[0]

        return (jnp.where(update,
                                new_x,
                                x),
                jnp.where(update,
                                new_chart,
                                chart),
                *ys)
    
    M.sde_Brownian_coords = sde_Brownian_coords
    M.chart_update_Brownian_coords = chart_update_Brownian_coords
    M.Brownian_coords = jit(lambda x,dts,dWs,stdCov=1.: integrate_sde(sde_Brownian_coords,integrator_ito,chart_update_Brownian_coords,x[0],x[1],dts,dWs,stdCov)[0:3])
