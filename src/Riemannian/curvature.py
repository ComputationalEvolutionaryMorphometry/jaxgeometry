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
from src.utils import *
def initialize(M):
    """ Riemannian curvature """

    """
    Riemannian Curvature tensor
    
    Args:
        x: point on manifold
    
    Returns:
        4-tensor R_ijk^l in with order i,j,k,l
        (see e.g. https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#(3,1)_Riemann_curvature_tensor )
        Note that sign convention follows e.g. Lee, Riemannian Manifolds.
    """
    M.R = jit(lambda x: -(jnp.einsum('pik,ljp->ijkl',M.Gamma_g(x),M.Gamma_g(x))
                -jnp.einsum('pjk,lip->ijkl',M.Gamma_g(x),M.Gamma_g(x))
                +jnp.einsum('likj->ijkl',M.DGamma_g(x))
                -jnp.einsum('ljki->ijkl',M.DGamma_g(x))))
    
    """
    Riemannian Curvature form
    R_u (also denoted Omega) is the gl(n)-valued curvature form u^{-1}Ru for a frame
    u for T_xM
    
    Args:
        x: point on manifold
    
    Returns:
        4-tensor (R_u)_ij^m_k with order i,j,m,k
    """
    M.R_u = jit(lambda x,u: jnp.einsum('ml,ijql,qk->ijmk',jnp.linalg.inv(u),R(x),u))
    
#    """
#    Sectional curvature
#    
#    Args:
#        x: point on manifold
#        e1,e2: two orthonormal vectors spanning the section
#    
#    Returns:
#        sectional curvature K(e1,e2)
#    """
#    @jit
#    def sec_curv(x,e1,e2):
#        Rflat = jnp.tensordot(M.R(x),M.g(x),[3,0])
#        sec = jnp.tensordot(
#                jnp.tensordot(
#                    jnp.tensordot(
#                        jnp.tensordot(
#                            Rflat, 
#                            e1, [0,0]), 
#                        e2, [0,0]),
#                    e2, [0,0]), 
#                e1, [0,0])
#        return sec
#    M.sec_curv = sec_curv
    
    """
    Ricci curvature
    
    Args:
        x: point on manifold
    
    Returns:
        2-tensor R_ij in order i,j
    """
    M.Ricci_curv = jit(lambda x: jnp.einsum('kijk->ij',M.R(x)))
    
    """
    Scalar curvature
    
    Args:
        x: point on manifold
    
    Returns:
        scalar curvature
    """
    M.S_curv = jit(lambda x: jnp.einsum('ij,ij->',M.gsharp(x),M.Ricci_curv(x)))
    
