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
from jaxgeometry.params import *
from jaxgeometry.utils import *


def initialize(M):
    """ Frame Bundle geometry """
    
    d  = M.dim

    def chart_update_FM(u,chart,*args):
        if M.do_chart_update != True:
            return (u,chart)
        
        x = (u[0:d],chart)
        nu = u[d:].reshape((d,-1))

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
        
        return (jnp.where(update,
                                jnp.concatenate((new_x,M.update_vector(x,new_x,new_chart,nu).flatten())),
                                u),
                jnp.where(update,
                                new_chart,
                                chart))
    M.chart_update_FM = chart_update_FM        

    #### Bases shifts, see e.g. Sommer Entropy 2016 sec 2.3
    # D denotes frame adapted to the horizontal distribution
    def to_D(u,w):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        wx = w[0:d]
        wnu = w[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        Dwx = wx
        Dwnu = jnp.tensordot(Gammanu,wx,(2,0))+wnu

        return jnp.concatenate((Dwx,Dwnu.flatten()))
    def from_D(u,Dw):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dwx = Dw[0:d]
        Dwnu = Dw[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        wx = Dwx
        wnu = -jnp.tensordot(Gammanu,Dwx,(2,0))+Dwnu

        return jnp.concatenate((wx,wnu.flatten())) 
        # corresponding dual space shifts
    def to_Dstar(u,p):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        px = p[0:d]
        pnu = p[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        Dpx = px-jnp.tensordot(Gammanu,pnu,((0,1),(0,1)))
        Dpnu = pnu

        return jnp.concatenate((Dpx,Dpnu.flatten()))
    def from_Dstar(u,Dp):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
        Dpx = Dp[0:d]
        Dpnu = Dp[d:].reshape((d,-1))        
    
        # shift to D basis
        Gammanu = jnp.tensordot(M.Gamma_g(x),nu,(2,0)).swapaxes(1,2)
        px = Dpx+jnp.tensordot(Gammanu,Dpnu,((0,1),(0,1)))
        pnu = Dpnu

        return jnp.concatenate((px,pnu.flatten()))
    M.to_D = to_D
    M.from_D = from_D
    M.to_Dstar = to_Dstar
    M.from_Dstar = from_Dstar
    
    ##### Horizontal vector fields:
    def Horizontal(u):
        x = (u[0][0:d],u[1])
        nu = u[0][d:].reshape((d,-1))
    
        # Contribution from the coordinate basis for x: 
        dx = nu
        # Contribution from the basis for Xa:
        Gammahgammaj = jnp.einsum('hji,ig->hgj',M.Gamma_g(x),nu) # same as Gammanu above
        dnu = -jnp.einsum('hgj,ji->hgi',Gammahgammaj,nu)

        return jnp.concatenate([dx,dnu.reshape((-1,nu.shape[1]))],axis=0)
    M.Horizontal = Horizontal
    

