'''
methods to generate and sample 2D and 3D discrete distributions assuming
linear interpolation of the data
'''

from moments import distribution_moments
import numpy as np
from numpy import pi, newaxis
from numpy.random import random_sample
from scipy.integrate import cumtrapz, trapz


def discrete_maxwellian3D(vx, vy, vz, mass=1.0, density=1.0,
                          bulk_velocity=[0.0, 0.0, 0.0], average_KE = 1.0):
    '''
    build a 3D maxwellian distribution given the number density, bulk velocity,
    and average kinetic energy per particle, sampled at the given velocities,
    allowing for bi/tri-maxwellian distribution (different KE in x, y, z)

    Parameters
    ----------
    vx, vy, vz : meshgrid-style numpy arrays, or 1D numpy array (all same)
        x, y, z velocities to sample the maxwellian
    mass : float
        mass of a particle
    density : float
        number density of the species
    bulk_velocity : 1x3 array-like of floats
        bulk velocity of the species
    average_KE : float or 1x3 array-like of floats
        average kinetic energy of a particle in the species
    
    Returns
    -------
    distribution : 3D numpy array with shape of velocities
        the maxwellian distribution evaluated at the x, y, z coordinates
    '''

    if vx.ndim is not 3:
        vx, vy, vz = np.meshgrid(vx, vy, vz, indexing='ij')

    average_KE = np.array(average_KE)
    if len(average_KE.shape) == 0 or len(average_KE) == 1:
        average_KE = average_KE.repeat(3)
    temperature = 2. / 3. * average_KE
    w2 = ((vx - bulk_velocity[0])**2 / temperature[0] + 
          (vy - bulk_velocity[1])**2 / temperature[1] +
          (vz - bulk_velocity[2])**2 / temperature[2])
    distribution = (density * (mass / (2 * pi))**(3./2.) /
                    temperature.prod()**(1./2.) * 
                    np.exp(-(mass / 2.) * w2))
    return distribution

def discrete_maxwellian3D_wrapper(distribution):
    vx = distribution._x
    vy = distribution._y
    vz = distribution._z
    f = distribution.distribution
    m = distribution.mass
    n = distribution.density
    u = distribution.momentum / m
    KE = distribution.kinetic_energy
    return discrete_maxwellian3D(vx, vy, vz, m, n, u, KE)

def equilibrium_maxwellian3D(n1, n2, u1, u2, T1, T2, m1, m2, vx1, vy1, vz1,
                             vx2, vy2, vz2, tau_ratio, return_all=False):
    '''
    get the equilibrium maxwellian given the densities, bulk velocities,
    temperatures, masses, and assuming n1/tau12 = n2/tau21

    Parameters
    ----------
    n1, n2 : floats
        number density of species 1 and 2
    u1, u2 : 1x3 numpy array floats
        bulk velocity of species 1 and 2
    T1, T2 : floats
        temperature (2/3 of average KE per particle) of species 1 and 2
    m1, m2 : floats
        mass of species 1 and 2
    vx*, vy*, vz* : meshgrid-style numpy arrays, or 1D arrays
        x, y, z velocities for the equilibrium distribution for each species
    tau_ratio : float
        ratio of taus such that tau_lk = tau_ratio * tau_kl
    return_all : boolean
        whether to return everything or just the equilibrium distributions

    Returns
    -------
    f12, f21 : 3D numpy array with shape of velocities
        the equilibrium distributions at the velocity coordinates
    u12 : 3x1 numpy array
        mixture bulk velocity
    T12 : float
        mixture temperature
    '''
    
    if vx1.ndim is not 3:
        vx1, vy1, vz1 = np.meshgrid(vx1, vy1, vz1, indexing='ij')
    if vx2.ndim is not 3:
        vx2, vy2, vz2 = np.meshgrid(vx2, vy2, vz2, indexing='ij')

    # mixture velocity
    u12 = ((n1 * m1 * tau_ratio * u1 + n2 * m2 * u2) /
           (n1 * m1 * tau_ratio + n2 * m2))

    # mixture temperature
    T12 = (3. * n1 * tau_ratio * T1 + 3. * n2 * T2 +
           n1 * m1 * tau_ratio * np.sum(u1**2 - u12**2) +
           n2 * m2 * np.sum(u2**2 - u12**2)) / (3. * n1 * tau_ratio + 3. * n2)

    # equilibrium distributions
    f12 = discrete_maxwellian3D(vx1, vy1, vz1, m1, n1, u12, 3./2. * T12)
    f21 = discrete_maxwellian3D(vx2, vy2, vz2, m2, n2, u12, 3./2. * T12)

    if not return_all:
        return f12, f21
    else:
        return f12, f21, u12, T12

def equilibrium_maxwellian3D_wrapper(distribution1, distribution2, tau_ratio):
    vx1 = distribution1._x
    vx2 = distribution2._x
    vy1 = distribution1._y
    vy2 = distribution2._y
    vz1 = distribution1._z
    vz2 = distribution2._z
    f1 = distribution1.distribution
    f2 = distribution2.distribution
    m1 = distribution1.mass
    m2 = distribution2.mass
    n1 = distribution1.density
    n2 = distribution2.density
    u1 = distribution1.momentum / m1
    u2 = distribution2.momentum / m2
    T1 = distribution1.kinetic_energy * 2./3.
    T2 = distribution2.kinetic_energy * 2./3.
    return equilibrium_maxwellian3D(n1, n2, u1, u2, T1, T2, m1, m2,
                                    vx1, vy1, vz1, vx2, vy2, vz2,
                                    tau_ratio, return_all=False)

class linear_interpolated_rv_2D(object):
    '''
    Contrtuct a 2D distribution assuming linear interpolation from discrete 
    data.
    '''

    def __init__(self, x, y, distribution):
        ''' intitialize the distribution object

        generates the necessary pdfs and cdfs to allow generation of samples

        Parameters
        ----------
        x : meshgrid, x-coordinates corresponding to the distribution
        y : meshgrid, y-coordinates corresponding to the distribution
        '''

        self._x = x[:,0]
        self._y = y[0,:]
        area_under_curve = trapz(trapz(distribution, axis=1), axis=0)
        self._pdf = distribution / area_under_curve
        self._ycdf = cumtrapz(self._pdf, axis=1, initial=0)
        self._ypdf = self._pdf / self._ycdf[:,-1][:,newaxis]
        self._ycdf /= self._ycdf[:,-1][:,newaxis]
        self._xpdf = trapz(self._pdf, axis=1)
        self._xcdf = cumtrapz(self._xpdf, axis=0, initial=0)
    

    def rvs(self, samples=1):
        ''' generate random samples from the distribution

        Parameters
        ----------
        samples : integer, number of random samples to draw

        Returns
        -------
        x : array with x-coordinates of each samples
        y : array with y-coordinates of each sample
        '''

        # first get the x-coordinates
        x_uniform = random_sample(samples)
        ix = self._xcdf.searchsorted(x_uniform)
        dx = ((-self._xpdf[ix-1] + np.sqrt(self._xpdf[ix-1]**2 +
            2 * (self._xpdf[ix] - self._xpdf[ix-1]) *
            (x_uniform - self._xcdf[ix-1]))) / 
            (self._xpdf[ix] - self._xpdf[ix-1]))
        x = self._x[ix-1] + dx * (self._x[ix] - self._x[ix-1])
        
        # interpolate the y-direction cdf and get y-coordinate
        interp_cdf = (self._ycdf[ix-1,:] * (1.0 - dx[:,newaxis]) +
                dx[:,newaxis] * (self._ycdf[ix,:]))
        interp_pdf = (self._ypdf[ix-1,:] * (1.0 - dx[:,newaxis]) +
                dx[:,newaxis] * (self._ypdf[ix,:]))
        y_uniform = random_sample(samples)
        jy = np.empty(y_uniform.shape, dtype=int)
        for j, sample in enumerate(y_uniform):
            jy[j] = interp_cdf[j,:].searchsorted(sample)
        iy = np.arange(0,samples)
        dy = ((-interp_pdf[iy,jy-1] + np.sqrt(interp_pdf[iy,jy-1]**2 +
            2 * (interp_pdf[iy,jy] - interp_pdf[iy,jy-1]) * 
            (y_uniform - interp_cdf[iy,jy-1]))) /
            (interp_pdf[iy,jy] - interp_pdf[iy,jy-1]))
        y = self._y[jy-1] + dy * (self._y[jy] - self._y[jy-1])
        return x, y


class linear_interpolated_rv_3D(object):
    '''
    Contrtuct a 3D distribution assuming linear interpolation from discrete 
    data.
    '''
    
    # initialization
    def __init__(self, x, y, z, distribution, mass=1):
        ''' intitialize the distribution object

        generates the necessary pdfs and cdfs to allow generation of samples

        Parameters
        ----------
        x : meshgrid or 1D array, x-coordinates for the distribution
        y : meshgrid or 1D array, y-coordinates for the distribution
        z : meshgrid or 1D array, z-coordinates for the distribution
        distribution : 3D array corresponding to distribution at velocities
        mass : mass of a particle in the species
        '''

        if x.ndim is not 3:
            x, y, z = np.meshgrid(x, y, z, indexing='ij')

        self._x = x[:,0,0]
        self._y = y[0,:,0]
        self._z = z[0,0,:]
        self.distribution = distribution

        # moments
        self.mass = mass
        (self.density, self.momentum, self.stress, self.kinetic_energy,
         self.heat, self.m4) = distribution_moments(x, y, z, distribution,
                                                    mass)

        # gradient of log(f)
        dx = x[1,0,0] - x[0,0,0]
        dy = y[0,1,0] - y[0,0,0]
        dz = z[0,0,1] - z[0,0,0]
        self.grad_log_f = np.stack(np.gradient(np.log(distribution),
                                               dx, dy, dx), axis=-1)

        
        # normalize the pdf
        area_under_curve = trapz(trapz(trapz(distribution, axis=2),
                                 axis=1), axis=0)
        self._pdf = distribution / area_under_curve
        
        # conditional pdf and cdf for the third dimension
        self._zcdf = cumtrapz(self._pdf, axis=2, initial=0)
        self._zpdf = self._pdf / self._zcdf[:,:,-1][:,:,newaxis]
        self._zcdf /= self._zcdf[:,:,-1][:,:,newaxis]
        
        # conditional pdf and cdf for the second dimention
        self._ypdf = trapz(self._pdf, axis=2)
        self._ycdf = cumtrapz(self._ypdf, axis=1, initial=0)
        self._ypdf /= self._ycdf[:,-1][:,newaxis]
        self._ycdf /= self._ycdf[:,-1][:,newaxis]
        
        # conditional pdf and cdf for the first dimension
        self._xpdf = trapz(trapz(self._pdf, axis=2), axis=1)
        self._xcdf = cumtrapz(self._xpdf, axis=0, initial=0)
    

    def rvs(self, samples=1):
        ''' generate random samples from the distribution

        Parameters
        ----------
        samples : integer, number of random samples to draw

        Returns
        -------
        samples x 3 array with x, y, z velocities of each sample
        '''

        # take only 100,000 samples at a time for memory reasons
        if samples > 100000:
            vel = []
            remaining = samples
            while remaining > 0:
                vel.append(self.rvs(min(remaining, 100000)))
                remaining -= 100000
            return np.vstack(vel)
        
        # first get the x-coordinates
        x_uniform = random_sample(samples)
        ix = self._xcdf.searchsorted(x_uniform)
        dx = ((-self._xpdf[ix-1] + np.sqrt(self._xpdf[ix-1]**2 +
            2 * (self._xpdf[ix] - self._xpdf[ix-1]) *
            (x_uniform - self._xcdf[ix-1]))) / 
            (self._xpdf[ix] - self._xpdf[ix-1]))
        # fix divisions by zero (e.g. if clipping of distribution)
        zerofix = np.isclose(self._xpdf[ix], self._xpdf[ix-1],
                             rtol=1e-12, atol=1e-12)
        dx[zerofix] = random_sample(zerofix.sum())
        x = self._x[ix-1] + dx * (self._x[ix] - self._x[ix-1])
        
        # interpolate the y-direction cdf and get y-coordinate
        interp_ycdf = (self._ycdf[ix-1,:] * (1.0 - dx[:,newaxis]) +
                dx[:,newaxis] * (self._ycdf[ix,:]))
        interp_ypdf = (self._ypdf[ix-1,:] * (1.0 - dx[:,newaxis]) +
                dx[:,newaxis] * (self._ypdf[ix,:]))
        y_uniform = np.random.random_sample(samples)
        jy = np.empty(y_uniform.shape, dtype=int)
        for j, sample in enumerate(y_uniform):
            jy[j] = interp_ycdf[j,:].searchsorted(sample)
        iy = np.arange(0,samples)
        dy = ((-interp_ypdf[iy,jy-1] + np.sqrt(interp_ypdf[iy,jy-1]**2 + 
            2 * (interp_ypdf[iy,jy] - interp_ypdf[iy,jy-1]) * 
            (y_uniform - interp_ycdf[iy,jy-1]))) /
            (interp_ypdf[iy,jy] - interp_ypdf[iy,jy-1]))
        # fix divisions by zero (e.g. if clipping of distribution)
        zerofix = np.isclose(interp_ypdf[iy,jy], interp_ypdf[iy,jy-1], 
                             rtol=1e-12, atol=1e-12)
        dy[zerofix] = np.random.random_sample(zerofix.sum())
        y = self._y[jy-1] + dy * (self._y[jy] - self._y[jy-1])
        
        # finally, interpolate in the z-direction
        interp_zcdf = ((1.0 - dx[:,newaxis] - dy[:,newaxis] + 
            dx[:,newaxis] * dy[:,newaxis]) * self._zcdf[ix-1,jy-1,:] +
            dx[:,newaxis] * (1.0 - dy[:,newaxis]) * self._zcdf[ix,jy-1,:] +
            dy[:,newaxis] * (1.0 - dx[:,newaxis]) * self._zcdf[ix-1,jy,:] +
            dx[:,newaxis] * dy[:,newaxis] * self._zcdf[ix,jy,:])
        interp_zpdf = ((1.0 - dx[:,newaxis] - dy[:,newaxis] + 
            dx[:,newaxis] * dy[:,newaxis]) * self._zpdf[ix-1,jy-1,:] +
            dx[:,newaxis] * (1.0 - dy[:,newaxis]) * self._zpdf[ix,jy-1,:] +
            dy[:,newaxis] * (1.0 - dx[:,newaxis]) * self._zpdf[ix-1,jy,:] +
            dx[:,newaxis] * dy[:,newaxis] * self._zpdf[ix,jy,:])
        z_uniform = random_sample(samples)
        kz = np.empty(z_uniform.shape, dtype=int)
        for k, sample in enumerate(z_uniform):
            kz[k] = interp_zcdf[k,:].searchsorted(sample)
        iz = np.arange(0,samples)
        dz = ((-interp_zpdf[iz,kz-1] + np.sqrt(interp_zpdf[iz,kz-1]**2 +
            2 * (interp_zpdf[iz,kz] - interp_zpdf[iz,kz-1]) * 
            (z_uniform - interp_zcdf[iz,kz-1]))) /
            (interp_zpdf[iz,kz] - interp_zpdf[iz,kz-1]))
        # fix divisions by zero (e.g. if clipping of distribution)
        zerofix = np.isclose(interp_zpdf[iz,kz], interp_zpdf[iz,kz-1],
                             rtol=1e-12, atol=1e-12)
        dz[zerofix] = random_sample(zerofix.sum())
        z = self._z[kz-1] + dz * (self._z[kz] - self._z[kz-1])
        
        return np.column_stack((x, y, z))


    def write_cross_section(self, filehandle):
        '''
        write a the x-direction cross section of the distribution,
        where y and z have been integrated out.
        '''
        
        np.save(filehandle, np.trapz(np.trapz(self.distribution, self._z, axis=2),
                                     self._y, axis=1))
