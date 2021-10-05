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

from jax.experimental import optimizers

# frame bundle most probable paths. System by Erlend Grong
# notation mostly follows Anisotropic covariance on manifolds and most probable paths,
# Erlend Grong and Stefan Sommer, 2021

def initialize(M):

    def ode_mpp(c,y):
        t,gammafvchi,chart = c
        lamb, = y
    
        lamb2 = lamb**2
        lambm2 = lamb**(-2)
        S = jnp.diag(lamb)
        invS = jnp.diag(1./lamb)

        gamma = gammafvchi[:M.dim] # point
        f = gammafvchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        gammaf = gammafvchi[:M.dim+M.dim**2] # point and frame
        v = gammafvchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # \dot{\gamma} in f-coordinates
        chi = jnp.zeros((M.dim,M.dim)) # \chi
        chi = chi.at[jnp.triu_indices(M.dim,1)].set(gammafvchi[2*M.dim+M.dim**2:])
        chi = chi.at[jnp.tril_indices(M.dim,-1)].set(-gammafvchi[2*M.dim+M.dim**2:])
        
        # derivatives
        R = jnp.einsum('stuv,si,tj,uk,vw,wl->ijkl',lax.stop_gradient(M.R((gamma,chart))),f,f,f,M.g((gamma,chart)),f) # curvature in f
        dv= .5*jnp.einsum('rs,sl,jikl,ij,k->r',S,S,R,chi,v)
        invSv = jnp.einsum('kl,l->k',invS,v)
        dchi = jnp.einsum('j,ik,k->ij',v,invS,invSv)-jnp.einsum('i,jk,k->ij',v,invS,invSv)
        #dv= .5*jnp.einsum('l,jikl,ij,k->l',lamb2,R,chi,v)
        #dchi = jnp.einsum('ij,i,j->ij',lambm2[:,None]-lambm2[None,:],v,v)
        dgammaf = jnp.einsum('ij,j->i',lax.stop_gradient(M.Horizontal((gammaf,chart))),v)
    
        return jnp.hstack((dgammaf.flatten(),dv,dchi[jnp.triu_indices(M.dim,1)]))
    
    def chart_update_mpp(gammafvchi,chart,*args):
        if M.do_chart_update is None:
            return (gammafvchi,chart)
    
        gamma = gammafvchi[:M.dim]
        f = gammafvchi[M.dim:M.dim+M.dim**2].reshape((M.dim,M.dim)) # frame
        v= gammafvchi[M.dim+M.dim**2:2*M.dim+M.dim**2] # anti-development of \dot{\gamma}
        chi = gammafvchi[2*M.dim+M.dim**2:] # \chi
        
        update = M.do_chart_update((gamma,chart))
        new_chart = M.centered_chart(M.F((gamma,chart)))
        new_gamma = M.update_coords((gamma,chart),new_chart)[0]
    
        return (jnp.where(update,
                                jnp.concatenate((new_gamma,M.update_vector((gamma,chart),new_gamma,new_chart,f).flatten(),v,chi)),
                                gammafvchi),
                jnp.where(update,
                                new_chart,
                                chart))
    
    
    M.mpp = jit(lambda gammafvchi,lamb,dts: integrate(ode_mpp,chart_update_mpp,dts,gammafvchi[0],gammafvchi[1],lamb))
    
    @jit
    def MPP_forwardt(u,lamb,v,chi,T=T,n_steps=n_steps):
        curve = M.mpp((jnp.hstack((u[0],v,chi)),u[1]),jnp.broadcast_to(lamb[None,...],(n_steps,)+lamb.shape),dts(T,n_steps))
        us = curve[1][:,0:M.dim+M.dim**2]
        vs = curve[1][:,M.dim+M.dim**2:2*M.dim+M.dim**2]
        chis = curve[1][:,2*M.dim+M.dim**2:]
        charts = curve[2]
        return(us,vs,chis,charts)
    M.MPP_forwardt = MPP_forwardt

    # optimization to satisfy end-point conditions
    # objective
    def MPP_f(vchi,lamb):
        v = vchi[0:M.dim]
        invlambv = v/lamb
        return jnp.dot(invlambv,invlambv)
    # constraint
    def MPP_c(vchi,u,lamb,y):
        v = vchi[0:M.dim]
        chi = vchi[M.dim:]
        xs,_,chis,charts = M.MPP_forwardt(u,lamb,v,chi)
        xT = xs[-1][0:M.dim]; chartT = charts[-1]; chiT = chis[-1]
        y_chartT = M.update_coords(y,chartT)
        #return (1./M.dim)*jnp.sum(jnp.square(xT-y_chartT[0]))+(2/(M.dim*(M.dim-1)))*jnp.sum(jnp.square(chiT))
        return jnp.hstack((xT-y_chartT[0],chiT))
    
    def MPP(u,lamb,y):
        res = scipy.optimize.minimize(MPP_f,
                        jnp.zeros(M.dim+M.dim*(M.dim-1)//2),
                        args=(lamb),
                        method='trust-constr',
                        constraints={'type':'eq','fun':MPP_c, 'args': (u,lamb,y)},
                        )
        vchi = res.x
        
        v = vchi[0:M.dim]
        chi = vchi[M.dim:]
        return (v,chi)
    M.MPP = MPP

    # mean and covariance (eigenvalues) computation
    # optimization
        
    # objective
    #     @jit
    def f(chart,x,lamb,v,chi):
    #     lamb /= np.prod(lamb)**(1/M.dim) # only determinant 1 relevant here
        invlambv = v/lamb
        return jnp.dot(invlambv,invlambv)+jnp.sum(jnp.log(lamb**2))
        
    # constraint
    #     @jit
    def _c(chart,x,lamb,v,chi,y,ychart):
        nu = jnp.linalg.cholesky(M.gsharp((x,chart)))
        u = (jnp.hstack((x,nu.flatten())),chart)
        xs,_,chis,charts = M.MPP_forwardt(u,lamb,v,chi)
        xT = xs[-1][0:M.dim]; chartT = charts[-1]; chiT = chis[-1]
        y_chartT = M.update_coords((y,ychart),chartT)
        return jnp.hstack((jnp.sqrt(M.dim)*(xT-y_chartT[0]),jnp.sqrt(2/(M.dim*(M.dim-1)))*chiT))
    def c(chart,x,lamb,v,chi,y,ychart):
        return jnp.sum(jnp.square(_c(chart,x,lamb,v,chi,y,ychart)))
    
    vg45_c = jit(jax.value_and_grad(c,(3,4)))
    jac2345_c = jit(jax.jacrev(_c,(1,2,3,4)))
    jac2345_f = jit(jax.value_and_grad(f,(1,2,3,4)))
    @jit
    def vg23_f(chart,x,lamb,v,chi,y,ychart):
        _jac2345_c = jac2345_c(chart,x,lamb,v,chi,y,ychart)
        invjac45_c = jnp.linalg.inv(jnp.hstack(_jac2345_c[2:4]))
        jac23_v = (-jnp.dot(invjac45_c,_jac2345_c[0]),-jnp.dot(invjac45_c,_jac2345_c[1])) # implicit function theorem
        jac23_v = (jac23_v[0][0:M.dim,:],jac23_v[1][0:M.dim,:])
    
        v_f, g_f = jac2345_f(chart,x,lamb,v,chi)
        g_f = (jnp.dot(g_f[2],jac23_v[0]),g_f[1]+jnp.dot(g_f[2],jac23_v[1]))
    
        return v_f, g_f

    def MPP_mean(x,chart,ys,step_size45=1e-1,step_size23=1e-3,num_steps=6000,opt23_update_mod=25):
        opt_init45, opt_update45, get_params45 = optimizers.adam(step_size45)
        opt_init23, opt_update23, get_params23 = optimizers.adam(step_size23)
    
        N = len(ys)
    
        def step(step, params, ys, opt_state23, opt_state45):
            params23 = get_params23(opt_state23); params45 = get_params45(opt_state45)
            value23 = 0.; grad23 = (jnp.zeros(M.dim),jnp.zeros(M.dim)); values45 = (); grads45 = ()
            for i in range(N):
                v45,g45 = vg45_c(*(params+params23+params45[i]+ys[i])) 
                values45 += (v45,); grads45 += (g45,)
            opt_state45 = opt_update45(step, grads45, opt_state45)
            if step % opt23_update_mod == 0:
                for i in range(N):
                    v23,g23 = vg23_f(*(params+params23+params45[i]+ys[i]))
                    value23 += 1/N*v23; grad23 = (grad23[0]+1/N*g23[0],grad23[1]+1/N*g23[1])        
                opt_state23 = opt_update23(step, grad23, opt_state23)
            return (value23, values45), (opt_state23, opt_state45)
    
        params = (chart,)
        params23 = (x[0],.5*np.ones(M.dim))
        params45 = ((jnp.zeros(M.dim),jnp.zeros(M.dim*(M.dim-1)//2)),)*N
        opt_state23 = opt_init23(params23)
        opt_state45 = opt_init45(params45)
    
        for i in range(num_steps):
            (value23, values45), (opt_state23, opt_state45) = step(i, params, ys, opt_state23, opt_state45)
            if i % 100 == 0:
                print("Step {} | T: {:0.6e} | T: {:0.6e} | T: {}".format(i, value23, jnp.max(jnp.array(values45)), str(get_params23(opt_state23)[1])))
        print("Step {} | T: {:0.6e} | T: {:0.6e} | T: {}".format(i, value23, jnp.max(jnp.array(values45)), str(get_params23(opt_state23)[1])))
    
        x,lamb = get_params23(opt_state23)
        vs,chis = zip(*get_params45(opt_state45))
        
        return (x,lamb,vs,chis)
    M.MPP_mean = MPP_mean

