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
from sklearn.decomposition import PCA

def tangent_PCA(M, Log, mean, y):
    y = list(y) # make sure y is subscriptable
    
    try:
        mpu.openPool()
        N = len(y)
        sol = mpu.pool.imap(lambda pars: (Log(mean,y[pars[0]],)[0],),mpu.inputArgs(range(N)))
        res = list(sol)
        Logs = mpu.getRes(res,0)
    except:
        mpu.closePool()
        raise
    else:
        mpu.closePool()

    pca = PCA()
    pca.fit(Logs)
    pca.transformed_Logs = pca.transform(Logs)

    return pca
