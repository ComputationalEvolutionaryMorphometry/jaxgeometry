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


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import lax
from jax import grad, jacfwd, jacrev, jit, vmap
from jax import random
from jax.scipy import optimize

from functools import partial

from src.utils import *

import time

from src.params import *

from multiprocess import Pool
import src.multiprocess_utils as mpu

import itertools
from functools import partial

