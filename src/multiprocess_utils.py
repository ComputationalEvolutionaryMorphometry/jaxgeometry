# # This file is part of Theano Geometry
#
# Copyright (C) 2017, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/theanogemetry
#
# Theano Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Theano Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#

###########################
#multiprocessing functions#
###########################

import dill
import numpy as np
from itertools import product
from functools import partial
from multiprocess import Pool
from multiprocess import cpu_count

pool = None

def openPool():
    global pool
    pool = Pool(cpu_count()//2)
    
def closePool():
    global pool
    pool.terminate()
    pool = None
    
def inputArgs(*args):
    return list(zip(*args))

def getRes(res,i):
    return np.array(list(zip(*res))[i])
