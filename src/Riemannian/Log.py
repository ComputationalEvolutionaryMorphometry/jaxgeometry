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

def initialize(M,f=None):
    """ numerical Riemannian Logarithm map """

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp
    def loss(x,v,y):
        (x1,chart1) = f(x,v)
        y_chart1 = M.update_coords(y,chart1)
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))
    dloss = jax.grad(loss,1)

    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
    def shoot(x,y,v0=None):        

        if v0 is None:
            v0 = jnp.zeros(M.dim)

        res = minimize(lambda w: (loss(x,w,y),dloss(x,w,y)), v0, method='BFGS', jac=True, options={'disp': False, 'maxiter': 100})

        return (res.x,res.fun)

    M.Log = shoot
