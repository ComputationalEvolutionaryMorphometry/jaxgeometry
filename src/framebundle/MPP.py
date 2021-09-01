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
        t,gammafvchi,chart = c
        Sigma,Sigmainv = y
    
        gamma = gammafvchi[:M.dim] # point
        f = gammafvchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        gammaf = gammafvchi[:M.dim+M.dim**2] # point and frame
        v = gammafvchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # \dot{\gamma} in f-coordinates
        chi = gammafvchi[2*M.dim+M.dim**2:].reshape((M.dim,M.dim)) # \chi
        
        # derivatives
        dv= .5*jnp.einsum('rl,ijkl,ij,k->r',Sigmainv,lax.stop_gradient(M.R((gamma,chart))),chi,v)
        Sigmav = jnp.einsum('ik,k->i',Sigma,v)
        dchi = jnp.einsum('j,i->ij',v,Sigmav)-jnp.einsum('i,j->ij',v,Sigmav)
        dgammaf = jnp.tensordot(lax.stop_gradient(M.Horizontal((gammaf,chart))),v,(1,0))
    
        return jnp.hstack((dgammaf.flatten(),dv,dchi.flatten()))
    
    def chart_update_mpp(gammafvchi,chart,*args):
        if M.do_chart_update is None:
            return (gammafvchi,chart)
    
        gamma = gammafvchi[:M.dim]
        f = gammafvchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        v= gammafvchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # anti-development of \dot{\gamma}
        chi = gammafvchi[2*M.dim+M.dim**2:].reshape((M.dim,M.dim)) # \chi
        
        update = M.do_chart_update((gamma,chart))
        new_chart = M.centered_chart(M.F((gamma,chart)))
        new_gamma = M.update_coords((gamma,chart),new_chart)[0]
    
        return (jnp.where(update,
                                jnp.concatenate((new_gamma,M.update_vector((gamma,chart),new_gamma,new_chart,f).flatten(),v,chi.flatten())),
                                gammafvchi),
                jnp.where(update,
                                new_chart,
                                chart))
    
    
    M.mpp = jit(lambda gammafvchi,Sigma,invSigma,dts: integrate(ode_mpp,chart_update_mpp,dts,gammafvchi[0],gammafvchi[1],Sigma,invSigma))
    
    def MPP_forwardt(u,Sigma,v,chi,T=T,n_steps=n_steps):
        curve = M.mpp((jnp.hstack((u[0],v,chi.flatten())),u[1]),jnp.broadcast_to(Sigma[None,...],(n_steps,)+Sigma.shape),jnp.broadcast_to(jnp.linalg.inv(Sigma)[None,...],(n_steps,)+Sigma.shape),dts(T,n_steps))
        us = curve[1][:,0:M.dim+M.dim**2]
        vs = curve[1][:,M.dim+M.dim**2:2*M.dim+M.dim**2]
        chis = curve[1][:,2*M.dim+M.dim**2:].reshape((-1,M.dim,M.dim))
        charts = curve[2]
        return(us,vs,chis,charts)
    M.MPP_forwardt = MPP_forwardt

    # optimization
    # objective
    def MPP_f(vchi,Sigma):
        v = vchi[0:M.dim]
        return jnp.dot(v,jnp.dot(Sigma,v))
    # constraint
    def MPP_c(vchi,u,Sigma,y):
        v = vchi[0:M.dim]
        chi = vchi[M.dim:].reshape((M.dim,M.dim)); chi = .5*(chi-chi.T)
        xs,_,chis,charts = M.MPP_forwardt(u,Sigma,v,chi)
        xT = xs[-1][0:M.dim]; chartT = charts[-1]; chiT = chis[-1]
        y_chartT = M.update_coords(y,chartT)
        return (1./M.dim)*jnp.sum(jnp.square(xT-y_chartT[0]))+(1./(M.dim**2-M.dim))*jnp.sum(jnp.square(chiT))
    
    def MPP(u,Sigma,y):
        res = scipy.optimize.minimize(MPP_f,
                        jnp.zeros(M.dim+M.dim**2),
                        args=(Sigma),
                        method='trust-constr',
                        constraints={'type':'eq','fun':MPP_c, 'args': (u,Sigma,y)},
                        )
        #print(res)
        vchi = res.x
        
        v = vchi[0:M.dim]
        chi = vchi[M.dim:].reshape((M.dim,M.dim)); chi = .5*(chi-chi.T)
        return (v,chi)
    M.MPP = MPP
