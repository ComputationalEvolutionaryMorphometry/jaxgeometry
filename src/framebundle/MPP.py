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

# frame bundle most probable paths. System by Erlend Grong
# notation mostly follows Anisotropic covariance on manifolds and most probable paths,
# Erlend Grong and Stefan Sommer, 2021

def initialize(M):

    def ode_mpp(c,y):
        t,gammaphivchi,chart = c
        lambd, = y
    
        gamma = gammaphivchi[:M.dim] # point
        phi = gammaphivchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        gammaphi = gammaphivchi[:M.dim+M.dim**2] # point and frame
        v = gammaphivchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # anti-development of \dot{\gamma}
        chi = gammaphivchi[2*M.dim+M.dim**2:].reshape((M.dim,M.dim)) # \chi
        
        # derivatives
        dv= .5*jnp.einsum('l,jikl,ij,k->l',lambd**2,lax.stop_gradient(M.R((gamma,chart))),chi,v)
        lambd2 = lambd**2; lambdm2 = lambd**(-2)
        dchi = jnp.einsum('ji,ij,i,j->ij',lambd2.reshape((2,1))-lambd2.reshape((1,2)),.5*jnp.outer(lambdm2,lambdm2),v,v)
        dgammaphi = jnp.dot(lax.stop_gradient(M.Horizontal((gammaphi,chart))),dv)
    
        return jnp.hstack((dgammaphi.flatten(),dv,dchi.flatten()))
    
    def chart_update_mpp(gammaphivchi,chart,*args):
        if M.do_chart_update is None:
            return (gammaphivphi,chart)
    
        gamma = gammaphivchi[:M.dim]
        phi = gammaphivchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        v= gammaphivchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # anti-development of \dot{\gamma}
        chi = gammaphivchi[2*M.dim+M.dim**2:].reshape((M.dim,M.dim)) # \chi
        
        update = M.do_chart_update((gamma,chart))
        new_chart = M.centered_chart(M.F((gamma,chart)))
        new_gamma = M.update_coords((gamma,chart),new_chart)[0]
    
        return (jnp.where(update,
                                jnp.concatenate((new_gamma,M.update_vector((gamma,chart),new_gamma,new_chart,phi).flatten(),v,chi.flatten())),
                                gammaphivchi),
                jnp.where(update,
                                new_chart,
                                chart))
    
    
    M.mpp = jit(lambda gammaphivchi,lambd,dts: integrate(ode_mpp,chart_update_mpp,dts,gammaphivchi[0],gammaphivchi[1],lambd))
    
    def MPPt(u,lambd,v,chi,T=T,n_steps=n_steps):
        curve = M.mpp((jnp.hstack((u[0],v,chi.flatten())),u[1]),jnp.tile(lambd,(n_steps,1)),dts(T,n_steps))
        us = curve[1][:,0:M.dim+M.dim**2]
        vs = curve[1][:,M.dim+M.dim**2:2*M.dim+M.dim**2]
        chis = curve[1][:,2*M.dim+M.dim**2:].reshape((-1,M.dim,M.dim))
        charts = curve[2]
        return(us,vs,chis,charts)
    M.MPPt = MPPt
