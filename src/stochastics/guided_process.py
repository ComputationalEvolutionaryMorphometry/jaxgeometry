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

#######################################################################
# guided processes, Delyon/Hu 2006                                    #
#######################################################################

# hit target v at time t=Tend
def get_guided(M,sde,chart_update,phi,sqrtCov=None,A=None,logdetA=None,method='DelyonHu',integration='ito'):
    """ guided diffusions """

    def sde_guided(c,y):
        t,x,chart,log_likelihood,log_varphi,T,v,*cy = c
        xchart = (x,chart)
        dt,dW = y
        
        (det,sto,X,*dcy) = sde((t,x,chart,*cy),y)
        
        h = jax.lax.cond(t<T-dt/2,
                         lambda _: phi(xchart,v,*cy)/(T-t),
                         lambda _: jnp.zeros_like(phi((x,chart),v,*cy)),
                         None)
        
        sto = jax.lax.cond(t < T-3*dt/2, # for Ito as well?
                           lambda _: sto,
                           lambda _: jnp.zeros_like(sto),
                           None)

        ### likelihood
        dW_guided = (1-.5*dt/(1-t))*dW+dt*h  # for Ito as well?
        sqrtCovx = sqrtCov(xchart,*cy) if sqrtCov is not None else X
        Cov = dt*jnp.tensordot(sqrtCovx,sqrtCovx,(1,1))
        Pres = jnp.linalg.inv(Cov)
        residual = jnp.tensordot(dW_guided,jnp.linalg.solve(Cov,dW_guided),(0,0))
        #residual = jnp.tensordot(dW_guided,jnp.tensordot(Pres,dW_guided,(1,0)),(0,0))
        log_likelihood = .5*(-dW.shape[0]*jnp.log(2*jnp.pi)-jnp.linalg.slogdet(Cov)[1]-residual)
        #log_likelihood = .5*(-dW.shape[0]*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(Pres)[1]-residual)

        ## correction factor
        ytilde = jnp.tensordot(X,h*(T-t),1)
        tp1 = t+dt
        if integration == 'ito':
            xtp1 = x+dt*(det+jnp.tensordot(X,h,1))+sto
        elif integration == 'stratonovich':
            tx = x+sto
            xtp1 = x+dt*det+.5*(sto+sde((tp1,tx,chart,*cy),y)[1])
        xtp1chart = (xtp1,chart)
        Xtp1 = sde((tp1,xtp1,chart,*cy),y)[2]
        ytildetp1 = jax.lax.stop_gradient(jnp.tensordot(Xtp1,phi(xtp1chart,v,*cy),1)) # to avoid NaNs in gradient

        # set default A if not specified
        Af = A if A != None else \
             lambda x,v,w,*args: jnp.dot(v,jnp.linalg.solve(jnp.tensordot(X,X,(1,1)),w))

        #     add t1 term for general phi
        #     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = jax.lax.cond(t<T-3*dt/2,
                          lambda _: -Af(xchart,ytilde,det*dt,*cy)/(T-t),
                          # check det term for Stratonovich (correction likely missing)
                          lambda _: 0.,
                          None)
        t34 = jax.lax.cond(tp1<T-3*dt/2,
                           lambda _: -(Af(xtp1chart,ytildetp1,ytildetp1,*cy)-Af(xchart,ytildetp1,ytildetp1,*cy)) / (
                           (T-tp1)),
                           lambda _: 0.,
                           None)
        log_varphi = t2 + t34

        return (det+jnp.dot(X,h),sto,X,log_likelihood,log_varphi,jnp.zeros_like(T),jnp.zeros_like(v),*dcy)
    
    def chart_update_guided(x,chart,log_likelihood,log_varphi,T,v,*ys):
        if chart_update is None:
            return (x,chart,log_likelihood,log_varphi,T,v,*ys)

        (x_new, chart_new, *ys_new) = chart_update(x,chart,*ys)
        v_new = M.update_coords((v,chart),chart_new)[0]
        return (x_new,chart_new,log_likelihood,log_varphi,T,v_new,*ys_new)
    
    guided = jit(lambda x,v,dts,dWs,*ys: integrate_sde(sde_guided,integrator_ito if integration == 'ito' else integrator_stratonovich,chart_update_guided,x[0],x[1],dts,dWs,0.,0.,jnp.sum(dts),M.update_coords(v,x[1])[0] if chart_update else v,*ys)[0:5])
   
    def _log_p_T(guided,A,phi,x,v,dW,dts,*ys):
        """ Monte Carlo approximation of log transition density from guided process """
        T = jnp.sum(dts)
        
        Cxv = jnp.sum(phi(x,M.update_coords(v,x[1])[0],*ys)**2)
        
        # sample
        log_varphis = jax.vmap(lambda dW: guided(x,v,dts,dW,*ys)[4][-1],1)(dW)
        
        log_varphi = jnp.log(jnp.mean(jnp.exp(log_varphis)))
        _logdetA = logdetA(x,*ys) if logdetA is not None else -2*jnp.linalg.slogdet(X)[1]
        log_p_T = .5*_logdetA-.5*x[0].shape[0]*jnp.log(2.*jnp.pi*T)-Cxv/(2.*T)+log_varphi
        return log_p_T
    log_p_T = partial(_log_p_T,guided,A,phi)

    neg_log_p_Ts = lambda *args: -jnp.mean(jax.vmap(lambda x,chart,w,dW,dts,*ys: log_p_T((x,chart),w,dW,dts,*ys),(None,None,0,0,None,*((None,)*(len(args)-5))))(*args))
    
        
    return (guided,sde_guided,chart_update_guided,log_p_T,neg_log_p_Ts)

