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

###############################################################
# Most probable paths for Kunita flows - BVP                  # 
###############################################################
def initialize(M,N):
    method='BFGS'

    def loss(x,v,y,qps,dqps,_dts):
        (_,xx1,charts) = M.MPP_AC(x,v,qps,dqps,_dts)
        (x1,chart1) = (xx1[-1,0],charts[-1])
        y_chart1 = M.update_coords(y,chart1)
        return 1./N.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))
    from scipy.optimize import approx_fprime
    dloss = lambda x,v,y,qps,dqps,_dts: approx_fprime(v,lambda v: loss(x,v,y,qps,dqps,_dts),1e-4)

    from scipy.optimize import minimize,fmin_bfgs,fmin_cg
    def shoot(x,y,qps,dqps,_dts,v0=None):        

        if v0 is None:
            v0 = jnp.zeros(N.dim)

        #res = minimize(jax.value_and_grad(lambda w: loss(x,w,y,qps,dqps,_dts)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})
        res = minimize(lambda w: (loss(x,w,y,qps,dqps,_dts),dloss(x,w,y,qps,dqps,_dts)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})
    #     res = minimize(lambda w: loss(x,w,y,qps,dqps,_dts), v0, method=method, jac=False, options={'disp': False, 'maxiter': 100})

    #     print(res)

        return (res.x,res.fun)

    M.Log_MPP_AC = shoot
