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

###############################################################
# Most probable paths for Kunita flows                        #
###############################################################
def initialize(M,N,sigmas,u):
    """ Most probable paths for Kunita flows                 """
    """ M: shape manifold, N: embedding space, u: flow field """
    
    # Riemannian structure on N
    N.gsharp = lambda x: jnp.einsum('pri,qrj->ij',sigmas(x[0]),sigmas(x[0]))
    delattr(N,'g')
    from src.Riemannian import metric
    metric.initialize(N)

    # scalar part of elliptic operator L = 1/2 \Delta_g + z
    z = lambda x,qp: (u(x,qp)
            -0.25*jnp.einsum('ij,i->j',N.gsharp(x),gradx(N.logAbsDetsharp)(x))
            -0.5*jnp.einsum('iji->j',jacrevx(N.gsharp)(x))
            +0.5*jnp.einsum('...rj,...rii->j',sigmas(x[0]),jax.jacrev(sigmas)(x[0]))
            )

    # Onsager-Machlup deviation from geodesic energy
    # f = lambda x,qp: .5*jnp.einsum('rs,sr->',N.gsharp(x),
    #                                    jacrevx(z)(x,qp)+jnp.einsum('k,srk->sr',z(x,qp),N.Gamma_g(x)))-1/12*N.S_curv(x)
    f = lambda x,qp: .5*N.divsharp(x,lambda x: z(x,qp))-1/12*N.S_curv(x)
    
    N.u = u
    N.z = z
    N.f = f

    def ode_MPP_AC(c,y):
        t,xx1,chart = c
        qp,dqp = y
        x = xx1[0] # point
        x1 = xx1[1] # derivative
        
        g = N.g((x,chart))
        gsharp = N.gsharp((x,chart))
        Gamma = N.Gamma_g((x,chart))
        
        zx = z((x,chart),qp)
        gradz = jacrevx(z)((x,chart),qp)
        dz = jnp.einsum('...ij,ij',jax.jacrev(z,argnums=1)((x,chart),qp),dqp)
        
        dx2 = (dz-jnp.einsum('i,j,kij->k',x1,x1,Gamma)
               +jnp.einsum('i,ki->k',x1,gradz+jnp.einsum('kij,j->ki',Gamma,zx))
               -jnp.einsum('rs,ri,s,ik->k',g,gradz+jnp.einsum('j,rij->ri',zx,Gamma),x1-zx,gsharp)
               +jnp.einsum('ik,i',gsharp,gradx(f)((x,chart),qp))
            )
        dx1 = x1
        return jnp.stack((dx1,dx2))

    def chart_update_MPP_AC(xv,chart,y):
        if M.do_chart_update is None:
            return (xv,chart)
    
        v = xv[1]
        x = (xv[0],chart)

        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                                jnp.stack((new_x,M.update_vector(x,new_x,new_chart,v))),
                                xv),
                jnp.where(update,
                                new_chart,
                                chart))
    
    M.MPP_AC = jit(lambda x,v,qps,dqps,dts: integrate(ode_MPP_AC,chart_update_MPP_AC,jnp.stack((x[0],v)),x[1],dts,qps,dqps))

