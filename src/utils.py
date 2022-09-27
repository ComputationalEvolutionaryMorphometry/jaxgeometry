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

from src.setup import *
from src.params import *

#######################################################################
# various useful functions                                            #
#######################################################################

# jax.grad but only for the x variable of a function taking coordinates and chart
def gradx(f):
    def fxchart(x,chart,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def gradf(x,*args,**kwargs):
        return jax.grad(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return gradf

# jax.jacfwd but only for the x variable of a function taking coordinates and chart
def jacfwdx(f):
    def fxchart(x,chart,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def jacf(x,*args,**kwargs):
        return jax.jacfwd(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

# jax.jacrev but only for the x variable of a function taking coordinates and chart
def jacrevx(f):
    def fxchart(x,chart,*args,**kwargs):
        return f((x,chart),*args,**kwargs)
    def jacf(x,*args,**kwargs):
        return jax.jacrev(fxchart,argnums=0)(x[0],x[1],*args,**kwargs)
    return jacf

# hessian only for the x variable of a function taking coordinates and chart
def hessianx(f):
    return jacfwdx(jacrevx(f))

# evaluation with pass through derivatives
def straight_through(f,x,*ys):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    if type(x) == type(()):
        zeros = tuple([xi - jax.lax.stop_gradient(xi) for xi in x])
        fx = jax.lax.stop_gradient(f(x,*ys))
        return tuple([fxi - jax.lax.stop_gradient(fxi) for fxi in fx])
    else:
        zero = x - jax.lax.stop_gradient(x)
        return zeros + jax.lax.stop_gradient(f(x,*ys))


# time increments, deterministic
def dts(T=T,n_steps=n_steps):
    return jnp.array([T/n_steps]*n_steps)
# standard noise realisations

# time increments, stochastic
seed = 42
global key
key = jax.random.PRNGKey(seed)
def dWs(d,_dts=None,num=1):
    global key
    keys = jax.random.split(key,num=num+1)
    key = keys[0]
    subkeys = keys[1:]
    if _dts == None:
        _dts = dts()
    if num == 1:
        return jnp.sqrt(_dts)[:,None]*random.normal(subkeys[0],(_dts.shape[0],d))
    else:
        return jax.vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*random.normal(subkey,(_dts.shape[0],d)))(subkeys)    

# Integrator (deterministic)
def integrator(ode_f,chart_update=None,method=default_method):
    if chart_update == None: # no chart update
        chart_update = lambda *args: args[0:2]

    # euler:
    def euler(c,y):
        t,x,chart = c
        dt,*_ = y
        return ((t+dt,*chart_update(x+dt*ode_f(c,y[1:]),chart,y[1:])),)*2

    # Runge-kutta:
    def rk4(c,y):
        t,x,chart = c
        dt,*_ = y
        k1 = ode_f(c,y[1:])
        k2 = ode_f((t+dt/2,x + dt/2*k1,chart),y[1:])
        k3 = ode_f((t+dt/2,x + dt/2*k2,chart),y[1:])
        k4 = ode_f((t,x + dt*k3,chart),y[1:])
        return ((t+dt,*chart_update(x + dt/6*(k1 + 2*k2 + 2*k3 + k4),chart,y[1:])),)*2

    if method == 'euler':
        return euler
    elif method == 'rk4':
        return rk4
    else:
        assert(False)

# return symbolic path given ode and integrator
def integrate(ode,chart_update,x,chart,dts,*ys):
    _,xs = lax.scan(integrator(ode,chart_update),
            (0.,x,chart),
            (dts,*ys))
    return xs if chart_update is not None else xs[0:2]

# sde functions should return (det,sto,Sigma) where
# det is determinisitc part, sto is stochastic part,
# and Sigma stochastic generator (i.e. often sto=dot(Sigma,dW)


def integrate_sde(sde,integrator,chart_update,x,chart,dts,dWs,*cy):
    _,xs = lax.scan(integrator(sde,chart_update),
            (0.,x,chart,*cy),
            (dts,dWs,))
    return xs

def integrator_stratonovich(sde_f,chart_update=None):
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler_heun(c,y):
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        tx = x + stox
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x + dt*detx + 0.5*(stox + sde_f((t+dt,tx,chart,*cy),y)[1]), chart, *cy_new),),)*2

    return euler_heun

def integrator_ito(sde_f,chart_update=None):
    if chart_update == None: # no chart update
        chart_update = lambda xp,chart,*cy: (xp,chart,*cy)

    def euler(c,y):
        t,x,chart,*cy = c
        dt,dW = y

        (detx, stox, X, *dcy) = sde_f(c,y)
        cy_new = tuple([y+dt*dy for (y,dy) in zip(cy,dcy)])
        return ((t+dt,*chart_update(x + dt*detx + stox, chart, *cy_new)),)*2

    return euler


def cross(a, b):
    return jnp.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]])

#import numpy as np
#def python_scan(f, init, xs, length=None):
#  if xs is None:
#    xs = [None] * length
#  carry = init
#  ys = []
#  for i in range(xs[0].shape[0]):
#    x = (xs[0][i],)
#    carry, y = f(carry, x)
#    ys.append(y)
#  return carry, np.stack(ys)
#
