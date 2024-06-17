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
from jaxgeometry.utils import *

def initialize(M,truncate_high_order_derivatives=False):
    """ add metric related structures to manifold """

    d = M.dim

    if hasattr(M, 'g'):
        if not hasattr(M, 'gsharp'):
            M.gsharp = lambda x: jnp.linalg.inv(M.g(x))
    elif hasattr(M, 'gsharp'):
        if not hasattr(M, 'g'):
            M.g = lambda x: jnp.linalg.inv(M.gsharp(x))
    else:
        raise ValueError('no metric or cometric defined on manifold')

    M.Dg = jacfwdx(M.g) # derivative of metric

    ##### Measure
    M.mu_Q = lambda x: 1./jnp.nlinalg.Det()(M.g(x))

    ### Determinant
    def det(x,A=None): 
        return jnp.linalg.det(M.g(x)) if A is None else jnp.linalg.det(jnp.tensordot(M.g(x),A,(1,0)))
    def detsharp(x,A=None): 
        return jnp.linalg.det(M.gsharp(x)) if A is None else jnp.linalg.det(jnp.tensordot(M.gsharp(x),A,(1,0)))
    M.det = det
    M.detsharp = detsharp
    def logAbsDet(x,A=None): 
        return jnp.linalg.slogdet(M.g(x))[1] if A is None else jnp.linalg.slogdet(jnp.tensordot(M.g(x),A,(1,0)))[1]
    def logAbsDetsharp(x,A=None): 
        return jnp.linalg.slogdet(M.gsharp(x))[1] if A is None else jnp.linalg.slogdet(jnp.tensordot(M.gsharp(x),A,(1,0)))[1]
    M.logAbsDet = logAbsDet
    M.logAbsDetsharp = logAbsDetsharp

    ##### Sharp and flat map:
    M.flat = lambda x,v: jnp.tensordot(M.g(x),v,(1,0))
    M.sharp = lambda x,p: jnp.tensordot(M.gsharp(x),p,(1,0))

    ##### Christoffel symbols
    # \Gamma^i_{kl}, indices in that order
    #M.Gamma_g = lambda x: 0.5*(jnp.einsum('im,mkl->ikl',M.gsharp(x),M.Dg(x))
    #               +jnp.einsum('im,mlk->ikl',M.gsharp(x),M.Dg(x))
    #               -jnp.einsum('im,klm->ikl',M.gsharp(x),M.Dg(x)))
    def Gamma_g(x):
        Dgx = M.Dg(x)
        gsharpx = M.gsharp(x)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    M.Gamma_g = Gamma_g
    M.DGamma_g = jacfwdx(M.Gamma_g)

    # Inner Product from g
    M.dot = lambda x,v,w: jnp.tensordot(jnp.tensordot(M.g(x),w,(1,0)),v,(0,0))
    M.norm = lambda x,v: jnp.sqrt(M.dot(x,v,v))
    M.norm2 = lambda x,v: M.dot(x,v,v)
    M.dotsharp = lambda x,p,pp: jnp.tensordot(jnp.tensordot(M.gsharp(x),pp,(1,0)),p,(0,0))
    M.conorm = lambda x,p: jnp.sqrt(M.dotsharp(x,p,p))

    ##### Gram-Schmidt and basis
    M.gramSchmidt = lambda x,u: (GramSchmidt_f(M.dotf))(x,u)
    M.orthFrame = lambda x: jnp.linalg.Cholesky(M.gsharp(x))

    ##### Hamiltonian
    M.H = lambda q,p: 0.5*jnp.tensordot(p,jnp.tensordot(M.gsharp(q),p,(1,0)),(0,0))

    # gradient, divergence, and Laplace-Beltrami
    M.grad = lambda x,f: M.sharp(x,gradx(f)(x))
    M.div = lambda x,X: jnp.trace(jacfwdx(X)(x))+.5*jnp.dot(X(x),gradx(M.logAbsDet)(x))
    M.divsharp = lambda x,X: jnp.trace(jacfwdx(X)(x))-.5*jnp.dot(X(x),gradx(M.logAbsDetsharp)(x))
    M.Laplacian = lambda x,f: M.div(x,lambda x: M.grad(x,f))
