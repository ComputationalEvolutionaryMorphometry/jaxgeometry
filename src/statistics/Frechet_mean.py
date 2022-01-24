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

from scipy.optimize import minimize

def Frechet_mean(Log, y, x0, options=None):
    y = list(y) # make sure y is subscriptable

    chart = x0[1]
        
    steps = []
    steps.append(x0[0])

    def fopts(x):
        N = len(y)
#        sol = mpu.pool.imap(lambda pars: (Log((x,chart),y[pars[0]])[0],),mpu.inputArgs(range(N)))
#        res = list(sol)
#        Logs = mpu.getRes(res,0)
        Logs = np.zeros((N, x.shape[0]))
        for i in range(N):
            Logs[i] = Log((x,chart), y[i])[0]

        res = (1. / N) * np.sum(np.square(Logs))
        grad = -(2. / N) * np.sum(Logs, 0)

        return (res, grad)

    def save_step(k):
        steps.append(k)

    try:
        mpu.openPool()
        res = minimize(fopts, x0[0], method='BFGS', jac=True, options=options, callback=save_step)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()

    return ((res.x,x0[1]), res.fun, np.array(steps))
