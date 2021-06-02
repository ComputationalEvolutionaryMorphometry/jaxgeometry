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

def initialize(M):
    """ Brownian motion from stochastic development """

    def Brownian_development(x,dWt):
        # amend x with orthogonal basis to get initial frame bundle element
        gsharpx = M.gsharp(x)
        nu = jnp.linalg.cholesky(gsharpx)
        u = (jnp.concatenate((x[0],nu.flatten())),x[1])
        
        (ts,us,charts) = M.stochastic_development(u,dWt)
        
        return (ts,us[:,0:M.dim],charts)
    
    M.Brownian_development = Brownian_development
