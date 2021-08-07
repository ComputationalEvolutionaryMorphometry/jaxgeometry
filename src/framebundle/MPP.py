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
    
        lambd2 = lambd**2; lambdm2 = lambd**(-2)

        gamma = gammaphivchi[:M.dim] # point
        phi = gammaphivchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        gammaphi = gammaphivchi[:M.dim+M.dim**2] # point and frame
        v = gammaphivchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # anti-development of \dot{\gamma}
        chi = gammaphivchi[2*M.dim+M.dim**2:].reshape((M.dim,M.dim)) # \chi
        
        # derivatives
        dv= .5*jnp.einsum('l,ijkl,ij,k->l',lambd2,lax.stop_gradient(M.R((gamma,chart))),chi,v)
        dchi = jnp.einsum('ji,ij,i,j->ij',lambd2.reshape((2,1))-lambd2.reshape((1,2)),.5*jnp.outer(lambdm2,lambdm2),v,v)
        dgammaphi = jnp.dot(lax.stop_gradient(M.Horizontal((gammaphi,chart))),v)
    
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
    
    def MPP_forwardt(u,lambd,v,chi,T=T,n_steps=n_steps):
        curve = M.mpp((jnp.hstack((u[0],v,chi.flatten())),u[1]),jnp.tile(lambd,(n_steps,1)),dts(T,n_steps))
        us = curve[1][:,0:M.dim+M.dim**2]
        vs = curve[1][:,M.dim+M.dim**2:2*M.dim+M.dim**2]
        chis = curve[1][:,2*M.dim+M.dim**2:].reshape((-1,M.dim,M.dim))
        charts = curve[2]
        return(us,vs,chis,charts)
    M.MPP_forwardt = MPP_forwardt

    # optimization
    # objective
    def f(vchi,u,lambd,y):
        v = vchi[0:M.dim]
        chi = vchi[M.dim:].reshape((M.dim,M.dim)); chi = .5*(chi-chi.T)
        xs,_,chis,charts = M.MPP_forwardt(u,lambd,v,chi)
        xT = xs[-1][0:M.dim]; chartT = charts[-1]; chiT = chis[-1]
        y_chartT = M.update_coords(y,chartT)
        return (1./M.dim)*jnp.sum(jnp.square(xT-y_chartT[0]))+(1./M.dim**2)*jnp.sum(jnp.square(chiT))
#    # constraint
#    def c(vchi,u,lambd):
#        v = vchi[0:M.dim]
#        chi = vchi[M.dim:].reshape((M.dim,M.dim)); chi = .5*(chi-chi.T)
#        xs,_,chis,charts = M.MPP_forwardt(u,lambd,v,chi)
#        chiT = chis[-1]
#        return 1e-8-(1./M.dim**2)*jnp.sum(jnp.square(chiT))
    
    def MPP(u,lambd,y):
        res = scipy.optimize.minimize(f,jnp.zeros(M.dim+M.dim**2),args=(u,lambd,y),method='BFGS',options={'disp': False, 'gtol': 1e-06, 'eps': 1e-4, 'maxiter': 100})
        #print(res)
        #res = scipy.optimize.minimize(f,
        #                res.x,
        #                args=(u,lambd,y),
        #                method='COBYLA',
        #                constraints={'type':'ineq','fun':c, 'args': (u,lambd)},
        #                )
        #print(res)
        vchi = res.x
        
        v = vchi[0:M.dim]
        chi = vchi[M.dim:].reshape((M.dim,M.dim)); chi = .5*(chi-chi.T)
        return (v,chi)
    M.MPP = MPP
