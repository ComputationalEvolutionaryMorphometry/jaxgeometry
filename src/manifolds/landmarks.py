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
from src.plotting import *

from src.manifolds.manifold import *

class landmarks(Manifold):
    """ LDDMM landmark manifold """

    def get_B(self,q):
        """ dual space basis for Laplacian kernel etc. """
        assert(not self.std_basis)

        PT = jnp.vstack((jnp.ones(self.N),q[0].reshape((self.N,self.m)).T))
        #codim = self.m*(self.N-PT.shape[0])
        #svd = jax.scipy.linalg.svd(PT)
        #V = svd[2].T[:,::-1] # reverse order
        V = jax.lax.linalg.eigh(PT.T@PT)[0]

        return jnp.kron(V,jnp.eye(self.m))

    def __init__(self,N=1,m=2,k_alpha=1.,k_sigma=None,kernel='Gaussian',order=2):
        Manifold.__init__(self)

        self.N = N # number of landmarks
        self.m = m # landmark space dimension (usually 2 or 3
        self.dim = self.m*self.N
        self.rank = self.dim

        # for cfg kernels
        self.std_basis = True
        self.order = order # order of Sobolev kernels
        self.codim = 0 # dimension of constraint null space

        self.update_coords = lambda coords,_: coords

        self.k_alpha = k_alpha
        if k_sigma == None:
            self.k_sigma = .5*jnp.eye(self.m)
        else:
            self.k_sigma = jnp.array(k_sigma) # standard deviation of the kernel
        self.inv_k_sigma = jnp.linalg.inv(self.k_sigma)
        self.k_Sigma = jnp.tensordot(self.k_sigma,self.k_sigma,(1,1))
        self.kernel = kernel

        ##### Kernel on M:
        if self.kernel == 'Gaussian':
            k = lambda x: self.k_alpha*jnp.exp(-.5*jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1))
        elif self.kernel == 'K0':
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                return self.k_alpha*jnp.exp(-r)
        elif self.kernel == 'K1':
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                return self.k_alpha*2*(1+r)*jnp.exp(-r)
        elif self.kernel == 'K2':
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                return self.k_alpha*4*(3+3*r+r**2)*jnp.exp(-r)
        elif self.kernel == 'K3':
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                return self.k_alpha*8*(15+15*r+6*r**2+r**3)*jnp.exp(-r)
        elif self.kernel == 'K4':
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                return self.k_alpha*16*(105+105*r+45*r**2+10*r**3+r**4)*jnp.exp(-r)
        elif self.kernel == 'laplacian':
            self.std_basis = False
            self.codim = self.m*int(scipy.special.binom(self.order-1+self.m,self.m))
            def k(x):
                r = jnp.sqrt((1e-7+jnp.square(jnp.tensordot(x,self.inv_k_sigma,(x.ndim-1,1))).sum(x.ndim-1)))
                if self.m % 2 == 0:
                    return self.k_alpha*(r**(2*self.order-self.m))*jnp.log(r)
                else:
                    return self.k_alpha*(r**(2*self.order-self.m))
        else:
            raise Exception('unknown kernel specified')
        self.k = k
        dk = lambda x: gradx(k)
        self.dk = dk
        d2k = lambda x: hessian(k)
        self.d2k = d2k

        # in coordinates
        self.k_q = lambda q1,q2: self.k(q1.reshape((-1,self.m))[:,np.newaxis,:]-q2.reshape((-1,self.m))[np.newaxis,:,:])
        self.K = lambda q1,q2: (self.k_q(q1,q2)[:,:,np.newaxis,np.newaxis]*jnp.eye(self.m)[np.newaxis,np.newaxis,:,:]).transpose((0,2,1,3)).reshape((-1,self.dim))

        ##### Metric:
        def gsharp(q):
            if self.std_basis:
                return self.K(q[0],q[0])
            else:
                B = self.get_B(q)
                Bkernel = B[:,:self.dim-self.codim]
                Bpoly = B[:,self.dim-self.codim:]
                gsharpB = jnp.einsum('ji,jk,kl->il',Bkernel,self.K(q[0],q[0]),Bkernel)
                gsharpextB = jax.scipy.linalg.block_diag((gsharpB),jnp.eye(self.codim))
                return jnp.einsum('ij,jk,kl->il',B,gsharpextB,jnp.linalg.inv(B))

        self.gsharp = gsharp

    
    def update_coords(self,coords,new_chart):
        return (coords[0],new_chart)

    def update_vector(self,coords,new_coords,new_chart,v):
        return v

    def update_covector(self,coords,new_coords,new_chart,p):
        return p

    ##### number of landmarks
    def setN(self, N):
        self.N = N # number of landmarks
        self.dim = self.m*self.N
        self.rank = self.dim

    ##### embedding space dimension
    def setm(self, m, k_sigma):
        self.m = m # landmark space dimension (usually 2 or 3
        self.dim = self.m*self.N
        self.rank = self.dim
        self.k_sigma = jnp.array(k_sigma) # standard deviation of the kernel
        self.inv_k_sigma = jnp.linalg.inv(self.k_sigma)

    #### k_sigma
    def setk_sigma(self,k_sigma):
        self.k_sigma = jnp.array(k_sigma) # standard deviation of the kernel
        self.inv_k_sigma = jnp.linalg.inv(self.k_sigma)
        self.k_Sigma = jnp.tensordot(self.k_sigma,self.k_sigma,(1,1))

    def __str__(self):
        return "%d landmarks in R^%d (dim %d). kernel %s, k_alpha=%d, k_sigma=%s, standard_basis=%s, cfg kernel codim=%d" % (self.N,self.m,self.dim,self.kernel,self.k_alpha,self.k_sigma,self.std_basis,self.codim)

    def newfig(self):
        if self.m == 2:
            newfig2d()
        elif self.m == 3:
            newfig3d()

    def plot(self):
        if self.m == 2:
            plt.axis('equal')

    def plot_path(self, xs, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, curve=False, markersize=None, arrowcolor='k'):
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            self.plotx(x, u=u if i == 0 else None,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       prevx=prevx,
                       last=i==N-1,
                       curve=curve)
            prevx = x
        return

    def plotx(self, x, u=None, color='b', color_intensity=1., linewidth=1., prevx=None, last=True, curve=False, markersize=None, arrowcolor='k'):
        assert(type(x) == type(()) or x.shape[0] == self.dim)
        if type(x) == type(()):
            (x,chart) = x
        if type(prevx) == type(()):
            (prevx,prevchart) = prevx
        #if not self.std_basis:
        #    x = self.get_B((x,chart))@x

        x = x.reshape((-1,self.m))
        NN = x.shape[0]

        ax = plt.gca()

        for j in range(NN):
            if last:
                if self.m == 2:
                    plt.scatter(x[j,0],x[j,1],color=color,s=markersize)
                elif self.m == 3:
                    ax.scatter(x[j,0],x[j,1],x[j,2],color=color,s=markersize if markersize else 50)
            else:
                try:
                    #if not self.std_basis:
                    #    prevx = self.get_B((prevx,prevchart))@prevx
                    prevx = prevx.reshape((NN,self.m))
                    xx = np.stack((prevx[j,:],x[j,:]))
                    if self.m == 2:
                        plt.plot(xx[:,0],xx[:,1],linewidth=linewidth,color=color)
                    elif self.m == 3:
                        ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
                except:
                    pass

            try:
                #if not self.std_basis:
                #    u = self.get_B((x,chart))@u
                u = u.reshape((NN, self.m))
                plt.quiver(x[j,0], x[j,1], u[j, 0], u[j, 1], pivot='tail', linewidth=linewidth, scale=5, color=arrowcolor)
            except:
                pass
        if curve and (last or prevx is None):
            plt.plot(np.hstack((x[:,0],x[0,0])),np.hstack((x[:,1],x[0,1])),'o-',color=color)


    # grid plotting functions
    import itertools

    """
    Example usage:
    (grid,Nx,Ny)=getGrid(-1,1,-1,1,xpts=50,ypts=50)
    plotGrid(grid,Nx,Ny)
    """

    def d2zip(self,grid):
        return np.dstack(grid).reshape([-1,2])

    def d2unzip(self,points,Nx,Ny):
        return np.array([points[:,0].reshape(Nx,Ny),points[:,1].reshape(Nx,Ny)])

    def getGrid(self,xmin,xmax,ymin,ymax,xres=None,yres=None,xpts=None,ypts=None):
        """
        Make regular grid
        Grid spacing is determined either by (x|y)res or (x|y)pts
        """

        if xres:
            xd = xres
        elif xpts:
            xd = np.complex(0,xpts)
        else:
            assert(False)
        if yres:
            yd = yres
        elif ypts:
            yd = np.complex(0,ypts)
        else:
            assert(False)

        grid = np.mgrid[xmin:xmax:xd,ymin:ymax:yd]
        Nx = grid.shape[1]
        Ny = grid.shape[2]

        return (self.d2zip(grid),Nx,Ny)


    def plotGrid(self,grid,Nx,Ny,coloring=True):
        """
        Plot grid
        """

        xmin = grid[:,0].min(); xmax = grid[:,0].max()
        ymin = grid[:,1].min(); ymax = grid[:,1].max()
        border = .5*(0.2*(xmax-xmin)+0.2*(ymax-ymin))

        grid = self.d2unzip(grid,Nx,Ny)

        color = 0.75
        colorgrid = np.full([Nx,Ny],color)
        cm = plt.cm.get_cmap('gray')
        if coloring:
            cm = plt.cm.get_cmap('coolwarm')
            hx = (xmax-xmin) / (Nx-1)
            hy = (ymax-ymin) / (Ny-1)
            for i,j in itertools.product(range(Nx),range(Ny)):
                p = grid[:,i,j]
                xs = np.empty([0,2])
                ys = np.empty([0,2])
                if 0 < i:
                    xs = np.vstack((xs,grid[:,i,j]-grid[:,i-1,j],))
                if i < Nx-1:
                    xs = np.vstack((xs,grid[:,i+1,j]-grid[:,i,j],))
                if 0 < j:
                    ys = np.vstack((ys,grid[:,i,j]-grid[:,i,j-1],))
                if j < Ny-1:
                    ys = np.vstack((ys,grid[:,i,j+1]-grid[:,i,j],))

                Jx = np.mean(xs,0) / hx
                Jy = np.mean(ys,0) / hy
                J = np.vstack((Jx,Jy,)).T

                A = .5*(J+J.T)-np.eye(2)
                CSstrain = np.log(np.trace(A*A.T))
                logdetJac = np.log(sp.linalg.det(J))
                colorgrid[i,j] = logdetJac

            cmin = np.min(colorgrid)
            cmax = np.max(colorgrid)
            f = 2*np.max((np.abs(cmin),np.abs(cmax),.5))
            colorgrid = colorgrid / f + 0.5

            print("mean color: ", np.mean(colorgrid))

        # plot lines
        for i,j in itertools.product(range(Nx),range(Ny)):
            if i < Nx-1:
                plt.plot(grid[0,i:i+2,j],grid[1,i:i+2,j],color=cm(colorgrid[i,j]))
            if j < Ny-1:
                plt.plot(grid[0,i,j:j+2],grid[1,i,j:j+2],color=cm(colorgrid[i,j]))

        #for i in range(0,grid.shape[1]):
        #    plt.plot(grid[0,i,:],grid[1,i,:],color)
        ## plot x lines
        #for i in range(0,grid.shape[2]):
        #    plt.plot(grid[0,:,i],grid[1,:,i],color)


        plt.xlim(xmin-border,xmax+border)
        plt.ylim(ymin-border,ymax+border)


    ### Misc
    def ellipse(self, cent, Amp):
        return  np.vstack(( Amp[0]*np.cos(np.linspace(0,2*np.pi*(1-1./self.N),self.N))+cent[0], Amp[1]*np.sin(np.linspace(0,2*np.pi*(1-1./self.N),self.N))+cent[1] )).T


