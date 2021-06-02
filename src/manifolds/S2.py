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
from src.params import *

from src.manifolds.ellipsoid import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

class S2(Ellipsoid):
    """ 2d Sphere """

    def __init__(self,use_spherical_coords=False,chart_center='z'):
        Ellipsoid.__init__(self,params=[1.,1.,1.],chart_center=chart_center,use_spherical_coords=use_spherical_coords)

    def __str__(self):
        return "%dd sphere (ellipsoid parameters %s, spherical_coords: %s)" % (self.dim,self.params,self.use_spherical_coords)

