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
# geodesic integration, Hamiltonian form                      #
###############################################################
def initialize(M):
    dq = grad(M.H,argnums=1)
    dp = lambda q,p: -gradx(M.H)(q,p)
    
    def ode_Hamiltonian(c,y):
        t,x,chart = c
        dqt = dq((x[0],chart),x[1])
        dpt = dp((x[0],chart),x[1])
        return jnp.stack((dqt,dpt))
    
    def chart_update_Hamiltonian(xp,chart,y):
        if M.do_chart_update is None:
            return (xp,chart)
    
        p = xp[1]
        x = (xp[0],chart)
    
        update = M.do_chart_update(x)
        new_chart = M.centered_chart(x)
        new_x = M.update_coords(x,new_chart)[0]
    
        return (jnp.where(update,
                            jnp.stack((new_x,M.update_covector(x,new_x,new_chart,p))),
                            xp),
                jnp.where(update,
                            new_chart,
                            chart))
    
    M.Hamiltonian_dynamics = jit(lambda q,p,dts: integrate(ode_Hamiltonian,chart_update_Hamiltonian,jnp.stack((q[0] if type(q)==type(()) else q,p)),q[1] if type(q)==type(()) else None,dts))
    
    def Exp_Hamiltonian(q,p,T=T,n_steps=n_steps):
        curve = M.Hamiltonian_dynamics(q,p,dts(T,n_steps))
        q = curve[1][-1,0]
        chart = curve[2][-1]
        return(q,chart)
    M.Exp_Hamiltonian = Exp_Hamiltonian
    def Exp_Hamiltoniant(q,p,T=T,n_steps=n_steps):
        curve = M.Hamiltonian_dynamics(q,p,dts(T,n_steps))
        qs = curve[1][:,0]
        charts = curve[2]
        return(qs,charts)
    M.Exp_Hamiltoniant = Exp_Hamiltoniant
