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
def get_guided(M,sde,chart_update,phi,sqrtCov=None,A=None,method='DelyonHu',integration='ito'):
    """ guided diffusions """

    def sde_guided(c,y):
        t,x,chart,log_likelihood,log_varphi,T,v,*cy = c
        xchart = (x,chart)
        dt,dW = y
        
        (det,sto,X,*dcy) = sde((t,x,chart,*cy),y)
        
        h = jax.lax.cond(t<T-dt/2,
                         lambda _: phi(xchart,v)/(T-t),
                         lambda _: jnp.zeros_like(phi((x,chart),v)),
                         None)
        
        sto = jax.lax.cond(t < T-3*dt/2, # for Ito as well?
                           lambda _: sto,
                           lambda _: jnp.zeros_like(sto),
                           None)

        ### likelihood
        dW_guided = (1-.5*dt/(1-t))*dW+dt*h  # for Ito as well?
        sqrtCovx = sqrtCov(xchart)
        Cov = dt*jnp.tensordot(sqrtCovx,sqrtCovx,(1,1))
        Pres = jnp.linalg.inv(Cov)
        residual = jnp.tensordot(dW_guided,jnp.tensordot(Pres,dW_guided,(1,0)),(0,0))
        log_likelihood = .5*(-dW.shape[0]*jnp.log(2*jnp.pi)+jnp.linalg.slogdet(Pres)[1]-residual)

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
        ytildetp1 = jnp.tensordot(Xtp1,phi(xtp1chart,v),1)

        # set default A if not specified
        Af = A if A != None else lambda x,v,w: jnp.dot(v,jnp.dot(jnp.linalg.inv(jnp.tensordot(X,X,(1,1))),w))

        #     add t1 term for general phi
        #     dxbdxt = theano.gradient.Rop((Gx-x[0]).flatten(),x[0],dx[0]) # use this for general phi
        t2 = jax.lax.cond(t<T-3*dt/2,
                          lambda _: -Af(xchart,ytilde,det*dt)/(T-t),
                          # check det term for Stratonovich (correction likely missing)
                          lambda _: 0.,
                          None)
        t34 = jax.lax.cond(tp1<T-3*dt/2,
                           lambda _: -(Af(xtp1chart,ytildetp1,ytildetp1)-Af(xchart,ytildetp1,ytildetp1)) / (
                           (T-tp1)),
                           lambda _: 0.,
                           None)
        log_varphi = t2 + t34

        return (det+jnp.dot(X,h),sto,X,log_likelihood,log_varphi,jnp.zeros_like(T),jnp.zeros_like(v),*dcy)
    
    def chart_update_guided(x,chart,log_likelihood,log_varphi,T,v,*ys):
        if chart_update is None:
            return (x, chart, T, v, *ys)

        (x_new, chart_new, *ys_new) = chart_update(x,chart,*ys)
        v_new = M.update_coords((v,chart),chart_new)[0]
        return (x_new, chart_new, log_likelihood, log_varphi, T, v_new, *ys_new)
    
    sde_guided = sde_guided
    guided = jit(lambda x,v,dts,dWs: integrate_sde(sde_guided,integrator_ito,chart_update_guided,x[0],x[1],dts,dWs,0.,0.,jnp.sum(dts),v)[0:5])
   
    return (guided,sde_guided)

#def bridge_sampling(lg,bridge_sdef,dWsf,options,pars):
#    """ sample samples_per_obs bridges """
#    (v,seed) = pars
#    if seed:
#        srng.seed(seed)
#    bridges = np.zeros((options['samples_per_obs'],n_steps.eval(),)+lg.shape)
#    log_varphis = np.zeros((options['samples_per_obs'],))
#    log_likelihoods = np.zeros((options['samples_per_obs'],))
#    for i in range(options['samples_per_obs']):
#        (ts,gs,log_likelihood,log_varphi) = bridge_sdef(lg,v,dWsf())[:4]
#        bridges[i] = gs
#        log_varphis[i] = log_varphi[-1]
#        log_likelihoods[i] = log_likelihood[-1]
#        try:
#            v = options['update_vf'](v) # update v, e.g. simulate in fiber
#        except KeyError:
#            pass
#    return (bridges,log_varphis,log_likelihoods,v)
#
## helper for log-transition density
#def p_T_log_p_T(g, v, dWs, bridge_sde, phi, options, F=None, sde=None, use_charts=False, chain_sampler=None, init_chain=None):
#    """ Monte Carlo approximation of log transition density from guided process """
#    if use_charts:
#        chart = g[1]
#    
#    # sample noise
#    (cout, updates) = theano.scan(fn=lambda x: dWs,
#                                  outputs_info=[T.zeros_like(dWs)],
#                                  n_steps=options['samples_per_obs'])
#    dWsi = cout
#    
#    # map v to M
#    if F is not None:
#        v = F(v if not use_charts else (v,chart))
#
#    if not 'update_v' in options:
#        # v constant throughout sampling
#        print("transition density with v constant")
#        
#        # bridges
#        Cgv = T.sum(phi(g, v) ** 2)
#        def bridge_logvarphis(dWs, log_varphi, chain):
#            if chain_sampler is None:
#                w = dWs
#            else:
#                (accept,new_w) = chain_sampler(chain)
#                w = T.switch(accept,new_w,w)
#            if not use_charts:
#                (ts, gs, log_likelihood, log_varphi) = bridge_sde(g, v, theano.gradient.disconnected_grad(w))[:4] # we don't take gradients of the sampling scheme
#            else:
#                (ts, gs, charts, log_likelihood, log_varphi) = bridge_sde(g, v, theano.gradient.disconnected_grad(w))[:5] # we don't take gradients of the sampling scheme
#            return (log_varphi[-1], w)
#
#        (cout, updates) = theano.scan(fn=bridge_logvarphis,
#                                      outputs_info=[constant(0.),init_chain if init_chain is not None else T.zeros_like(dWs)],
#                                      sequences=[dWsi])
#        log_varphi = T.log(T.mean(T.exp(cout[0])))
#        log_p_T = -.5 * g[0].shape[0] * T.log(2. * np.pi * Tend) - Cgv / (2. * Tend) + log_varphi
#        p_T = T.exp(log_p_T)
#    else:
#        # update v during sampling, e.g. for fiber densities
#        assert(chain_sampler is None)
#        print("transition density with v updates")
#
#        # bridges
#        def bridge_p_T(dWs, lp_T, lv):
#            Cgv = T.sum(phi(g, lv) ** 2)
#            (ts, gs, log_likelihood, log_varphi) = bridge_sde(g, lv, dWs)[:4]
#            lp_T =  T.power(2.*np.pi*Tend,-.5*g[0].shape[0])*T.exp(-Cgv/(2.*Tend))*T.exp(log_varphi[-1])
#            lv = options['update_v'](lv)                        
#            return (lp_T, lv)
#
#        (cout, updates) = theano.scan(fn=bridge_p_T,
#                                      outputs_info=[constant(0.), v],
#                                      sequences=[dWsi])
#        p_T = T.mean(cout[:][0])
#        log_p_T = T.log(p_T)
#        v = cout[-1][1]
#    
#    if chain_sampler is None:
#        return (p_T,log_p_T,v)
#    else:
#        return (p_T,log_p_T,v,w)
#
## densities wrt. the Riemannian volume form
#def p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[0]
#def log_p_T(*args,**kwargs): return p_T_log_p_T(*args,**kwargs)[1]
#
#def dp_T(thetas,*args,**kwargs):
#    """ Monte Carlo approximation of transition density gradient """
#    lp_T = p_T(*args,**kwargs)
#    return (lp_T,)+tuple(T.grad(lp_T,theta) for theta in thetas)
#
#def dlog_p_T(thetas,*args,**kwargs):
#    """ Monte Carlo approximation of log transition density gradient """
#    llog_p_T = log_p_T(*args,**kwargs)
#    return (llog_p_T,)+tuple(T.grad(llog_p_T,theta) for theta in thetas)
