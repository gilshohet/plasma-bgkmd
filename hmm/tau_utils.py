''' Compute the entropy production rate and relaxation parameters from MD data

Reads in the output from an MD simulation and computes dHdt, assuming f_k
remains constant throughout the simulation. Using  dHdt values, computes
tau_kl and feq_kl pairs using a Newton solve.
'''

import tau_helpers
import math
import numpy as np
from scipy.interpolate import interpn
import logging

def compute_dHdt(params, distribution, vels, forces):
    ''' compute dHdt from the MD data

    Parameters
    ----------
    params : md_parameters object
        all the simulation parameters
    distribution : list of distribution objects
        distributions for each species
    vels : n_particles x 3 numpy array
        particle velocities
    forces : n_particles x 3*n_species numpy array
        forces on each particle from each species
    

    Returns
    -------
    dHdt : n_species x n_species numpy array 
        dHdt[k,l] -> rate of change of H of species k due to species l
    '''

    # initialize stuff
    dHdt = np.empty((params.n_species, params.n_species))
    sp_end = np.cumsum(params.particles)
    sp_start = sp_end - params.particles

    # loop over first species (k)
    for sp_k in np.arange(params.n_species):
        # extract velocities, forces, and distribution info
        vel = vels[sp_start[sp_k]:sp_end[sp_k],:]
        force = forces[sp_start[sp_k]:sp_end[sp_k],:]
        dist = distribution[sp_k].distribution 
        vx = distribution[sp_k]._x
        vy = distribution[sp_k]._y
        vz = distribution[sp_k]._z
        dv = [vx[1]-vx[0], vy[1]-vy[0], vz[1]-vz[0]]
        mass = distribution[sp_k].mass

        # compute gradients
        grad_v_log_f = distribution[sp_k].grad_log_f
        
        # loop over other species (l)
        for sp_l in np.arange(params.n_species):
            # extract force from species l on particles in species k
            force_lk = force[:,sp_l*3:(sp_l+1)*3]

            # interpolate gradient of log of distribution at velocities
            grad_at_v = interpn((vx, vy, vz), grad_v_log_f, vel,
                                method='linear', bounds_error=False)

            # compute dot product of forces with gradients (this is dHdt)
            dHdt[sp_k,sp_l] = np.sum(force_lk / mass * grad_at_v)

    volume = (params.cell_size)**3
    dHdt /= volume

    return dHdt


def compute_taus(params, distribution, dHdt):
    '''compute all the taus given the distributions and dHdt
    
    computes intraspecies taus, then computes interspecies pairwise by solving
    a nonlinear least squares problem

    Parameters
    ----------
    params : md_parameters object
        all the relevant simulation parameters
    distribution : list of distribution objects
        distributions for each species
    dHdt : n_species x n_species numpy array of floats
        average dHdt of each species due to each other species
    method : string, either momentum or temperature
        whether to use momentum transfer or temperature relaxation collisions
    
    Returns
    -------
    taus : n_species x n_species numpy array of floats
        all of the intra- and inter-species relaxation timescales
    '''
    
    taus = np.empty((params.n_species, params.n_species))
    error = np.zeros((params.n_species, params.n_species))

    # first compute the intraspecies taus
    for sp in range(params.n_species):
        taus[sp,sp] = tau_helpers.compute_intraspecies_tau(distribution[sp],
                                                           dHdt[sp,sp])

    # now loop over species pairs to compute interspecies taus
    for sp1 in range(params.n_species):
        for sp2 in range(sp1+1, params.n_species):
            tau12, tau21, err = tau_helpers.compute_interspecies_tau(
                    distribution[sp1], distribution[sp2], dHdt[sp1,sp2],
                    dHdt[sp2,sp1])
            taus[sp1,sp2] = tau12
            taus[sp2,sp1] = tau21
            error[sp1,sp2] = err[0]
            error[sp2,sp1] = err[1]

    # if intraspecies taus essentially zero, use analytical form
    analytical_taus = momentum_transfer_tau(params)
    logging.debug('temperature relaxation taus are:\n' + np.array_str(analytical_taus))
    logging.debug('computed taus from md are:\n' + np.array_str(taus))
    for sp in range(params.n_species):
        if 1: #taus[sp,sp] < 1e-2:
            taus[sp,sp] = analytical_taus[sp,sp]

    return taus, error


def momentum_transfer_tau(params):
    ''' compute an estimate of the relaxation rate based on the momentum
    transfer collision rate derived using Stanton-Murillo cross-sections and
    modified to allow for multiple temperatures

    Parameters
    ----------
    params : md_parameters object
        all the parameters of the species in question

    Returns
    -------
    taus : n_species x n_species numpy array of floats
        all of the intra- and inter-species relaxation timescales
    '''

    # setup
    n_species = params.n_species
    density = params.density
    mass = params.mass
    temp = 2./3. * params.kinetic_energy
    charge = params.charge
    screen = params.screen_length
    taus = np.empty((n_species, n_species))
    a = np.array([0., 1.4660, -1.7836, 1.4313, -0.55833, 0.06112])
    b = np.array([0.081033, -0.091336, 0.051760, -0.50026, 0.17044])

    # loop over species pairs
    for sp1 in range(n_species):
        for sp2 in range(n_species):
            K_ij = mass[sp1] * mass[sp2] / (2. * (mass[sp1] * temp[sp2] +
                                                  mass[sp2] * temp[sp1]))
            mu_ij = mass[sp1] * mass[sp2] / (mass[sp1] + mass[sp2])
            gamma_ij = 2 * charge[sp1] * charge[sp2] * K_ij / (screen * mu_ij)
            if gamma_ij < 1:
                K_11 = (-1./4. * 
                        math.log(sum([a[k] * gamma_ij**k for k in range(5)])))
            else:
                K_11 = (b[0] + b[1] * math.log(gamma_ij) + b[2] * 
                        math.log(gamma_ij)**2) / (1 + b[3] * gamma_ij +
                                                  b[4] * gamma_ij**2)
            taus[sp1,sp2] = ((density[sp1] * mass[sp1] * 3. * 
                    (2. * np.pi)**(3./2.) * mu_ij *
                    (mass[sp2] * temp[sp1] + mass[sp1] * temp[sp2])**(3./2.)) /
                    (128. * np.pi**2 * density[sp1] * density[sp2] *
                    (mass[sp1] * mass[sp2])**(3./2.) *
                    (charge[sp1] * charge[sp2])**2 * K_11))

    return taus

def temperature_relaxation_taus(params):
    ''' compute an estimate of the relaxation rate based on the temperature
    relastion collision rate derived using Stanton-Murillo cross-sections and
    modified to allow for multiple temperatures

    Parameters
    ----------
    params : md_parameters object
        all the parameters of the species in question

    Returns
    -------
    taus : n_species x n_species numpy array of floats
        all of the intra- and inter-species relaxation timescales
    '''

    mass = params.mass
    taus = ((mass[:,np.newaxis] + mass[np.newaxis,:]) /
            (2. * mass[:,np.newaxis]) * momentum_transfer_tau(params))
    
    return taus

