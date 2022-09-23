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

from jax.example_libraries import optimizers

def iterative_mle(obss,neg_log_p_Ts,params,params_inds,params_update,chart,_dts,M,N=1,step_size=1e-1,num_steps=50):
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    vg = jax.value_and_grad(neg_log_p_Ts,params_inds)

    def step(step, params, opt_state, chart):
        params = get_params(opt_state)
        value,grads = vg(params[0],chart,obss,dWs(len(obss[0])*N*M.dim,_dts).reshape(-1,_dts.shape[0],N,M.dim),_dts,*params[1:])
        opt_state = opt_update(step, grads, opt_state)
        opt_state,chart = params_update(opt_state, chart)
        return (value,opt_state,chart)

    opt_state = opt_init(params)
    values = (); paramss = ()

    for i in range(num_steps):
        (value, opt_state, chart) = step(i, params, opt_state, chart)
        values += (value,); paramss += ((*get_params(opt_state),chart),)
        if i % 1 == 0:
            print("Step {} | T: {:0.6e} | T: {}".format(i, value, str((get_params(opt_state),chart))))
    print("Final {} | T: {:0.6e} | T: {}".format(i, value, str(get_params(opt_state))))
    
    return (get_params(opt_state),chart,value,jnp.array(values),paramss)
