''' set MD parameters, run the simulation, and read data files

Given the MD conditions, write the parameter file, particle positions and
velocities, resample velocities, and start the simulation given the parameters.
'''

import tau_utils
import distributions
from moments import particle_moments
import os
import logging
import numpy as np
from numpy.random import uniform, multivariate_normal

#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

class md_parameters(object):
    ''' class with all the parameters for an MD simulation

    EVERYTHING IN ATOMIC UNITS

    n_sims : int
        number of simulations to run
    cell_size : float
        length of each side of computational box as multiple of screen_length
    n_timesteps : int
        number of timesteps for the main simulation
    timestep : float
        size of each time step in terms of shortest plasma period (1 / f_p)
    movie_rate : int
        number of time steps between outputting data in ovito format
    tau_output_rate : int
        number of time steps between output data for computing tau
    n_species : int
        number of species
    density : n_species length array of floats
        density in bohr**-3 of each species
    mass : n_species length array-like of floats
        mass of a particle for each species
    charge : n_species length array-like of floats
        charge of a particle for each species
    screen_length : float
        the electron screening length
    cutoff : float
        cutoff * screen_length = cutoff radius for computing interactions
    ext_field : 3x1 array-like of floats
        the external electric field
    equilibration_time : float
        length of equilibration in terms of the slowest plasma period (1 / f_p)
        (actual time is double this because using strong then weak thermostat)
    stress : n_species x 3 x 3 array-like float
        stress tensor for the species, used in equilibration
    bulk_velocity : nSpecies x 3 array-like of floats
        bulk velocity of for equlibration for each species
    friction : n_species or 1 length array-like of floats
        equilibration friction of species as factor of plasma frequency (w_p)
    n_mts_timesteps : int
        number of timesteps to take in a normal time step for MTS
    mts_cutoff : float
        distance of cutoff for particles to be labeled MTS in screening lengths
    mts_threshold : float
        relative change in total energy to trigger MTS step
    n_procs : int
        number of cores to use for MD simulations, default value will use all
    '''

    def __init__(self, n_sims=1, cell_size=20.0, n_timesteps=1000,
                 timestep=1.0e-3, movie_rate=10, n_species=1, density=[1.],
                 mass=[1.], charge=[1.0], screen_length=1.0e-1, cutoff=10.,
                 ext_field=[0.0, 0.0, 0.0], equilibration_time=1.0,
                 stress=np.eye(3), bulk_velocity=[[0.0, 0.0, 0.0]],
                 friction=[1.0], n_mts_timesteps=100, mts_cutoff=1.0,
                 mts_threshold=100, n_procs=1000000):
        self.n_sims = n_sims
        self.cell_size = cell_size * screen_length
        self.n_timesteps = n_timesteps
        self.movie_rate = movie_rate
        self.n_species = n_species
        self.density = np.array(density)
        self.mass = np.array(mass)
        self.charge = np.array(charge)
        self.screen_length = screen_length
        self.cutoff = cutoff
        self.ext_field = ext_field
        self.equilibration_time = equilibration_time
        self.stress = stress
        self.bulk_velocity = np.array(bulk_velocity)
        self.n_mts_timesteps = n_mts_timesteps
        self.mts_cutoff = mts_cutoff
        self.mts_threshold = mts_threshold
        self.n_procs = n_procs

        # get plasma_frequency (f_p), timestep, and friction
        self.plasma_frequency = np.sqrt((self.density * self.charge**2) /
                                        (self.mass * np.pi))
        self.timestep = timestep / self.plasma_frequency.max()
        self.friction = np.array(friction) * self.plasma_frequency * 2. * np.pi

        # get number of particles
        self.particles = (density * (cell_size * screen_length)**3).astype(int)
        self.n_particles = self.particles.sum()

    def change_cell_size(self, cell_size):
        ''' change the cell size and recompute number of particles

        Parameters
        ----------
        cell_size : float
            the new cell size in multiples of the screening length
        '''

        self.cell_size = cell_size * self.screen_length
        self.particles = (self.density *
                          (cell_size * self.screen_length)**3).astype(int)
        self.n_particles = self.particles.sum()


    def change_timestep(self, timestep, md):
        ''' change the timestep for the simulation

        Parameters
        ----------
        timestep : float
            timestep in terms of fastest plasma period
        md : md module
            the md simulation
        '''

        self.timestep = timestep / self.plasma_frequency.max()
        md.parameters_mod.timestep = self.timestep
        md.parameters_mod.halftimestep = 0.5 * self.timestep

    def change_friction(self, friction, md):
        ''' change the friction parameter in the simulation

        Parameters
        ----------
        friction : float or n_species array of floats
            the friction in terms of plasma_frequency (w_p)
        md : md module
            the md simulation
        '''

        self.friction = np.array(friction) * self.plasma_frequency * 2. * np.pi

        # change friction in md module
        sp_end = np.cumsum(self.particles)
        sp_start = sp_end - self.particles
        for sp in range(self.n_species):
            md.particles_mod.friction[sp_start[sp]:sp_end[sp]] = \
                    self.friction[sp]


class md_data(object):
    ''' object for storing the processed output of a given MD simulation,
    including the instantaneous moments and dHdt

    bulk_velocity : n_sims x n_timesteps+1 x n_species x 3 numpy array
        the x,y,z bulk_velocity for each species at each time
    momentum : n_sims x n_timesteps+1 x n_species x 3 numpy array
        the x,y,z momentum per particle for each species at each time
    stress : n_sims x n_timesteps+1 x n_species x 3 x 3 numpy array
        the stress tensor for each species at each time
    kinetic_energy : n_sims x n_timesteps+1 x n_species numpy array
        the kinetic energy per particle for each species at each time
    heat : n_sims x n_timesteps+1 x n_species x 3 numpy array
        the x,y,z heat transfer per particle for each species at each time
    m4 : n_sims x n_timesteps+1 x n_species numpy array
        fourth moment of the distribution for each species at each time
    dHdt : n_sims x n_timesteps+1 x n_species x n_species numpy array
        dHdt of each species due to each other species at each time
    mass : n_species numpy array
        mass of each species
    time : n_timesteps numpy array
        physical simulation time at each save point
    '''

    def __init__(self, params):
        ''' load the data and generate the data object

        Parameters
        ----------
        params : md_parameters object
            all the simulation parameters
        '''

        # initialize stuff
        self.mass = params.mass
        self.time = (params.timestep * np.arange(params.n_timesteps+1))
        self.dHdt = np.empty((params.n_sims, params.n_timesteps+1,
                              params.n_species, params.n_species))
        self.momentum = np.empty((params.n_sims, params.n_timesteps+1,
                                  params.n_species, 3))
        self.velocity = np.empty((params.n_sims, params.n_timesteps+1,
                                  params.n_species, 3))
        self.stress = np.empty((params.n_sims, params.n_timesteps+1,
                                params.n_species, 3, 3))
        self.kinetic_energy = np.empty((params.n_sims, params.n_timesteps+1,
                                        params.n_species))
        self.heat = np.empty((params.n_sims, params.n_timesteps+1,
                              params.n_species, 3))
        self.m4 = np.empty((params.n_sims, params.n_timesteps+1,
                            params.n_species))


    def get_md_data(self, sim, step, params, md, distribution):
        ''' compute the particle moments and dHdt from the MD simulation
        particle data

        Parameters
        ----------
        sim : int
            which simulation number we are on (zero indexed)
        step : int
            which time step we are on
        params : md_parameters object
            the simulation parameters
        md : md module
            the md simulation
        distritubion : array of distribution objects
            the distributions for evaluating stuff for dhdt
        '''

        # extract positions and velocities
        pos, vel = get_md_phasespace(md)

        # extract forces
        forces = get_md_forces(params, md)

        # compute moments
        sp_end = np.cumsum(params.particles)
        sp_start = sp_end - params.particles
        for sp in range(params.n_species):
            (momentum, stress, kinetic_energy, heat, m4) = \
                    particle_moments(vel[sp_start[sp]:sp_end[sp],:],
                                     self.mass[sp])
            velocity = momentum / self.mass[sp]

            # fill class variables
            self.momentum[sim,step,sp,:] = momentum
            self.stress[sim,step,sp,:,:] = stress
            self.kinetic_energy[sim,step,sp] = kinetic_energy
            self.heat[sim,step,sp,:] = heat
            self.m4[sim,step,sp] = m4
            self.velocity[sim,step,sp,:] = velocity

        # compute dHdt
        dHdt = tau_utils.compute_dHdt(params, distribution, vel, forces)
        self.dHdt[sim,step,:,:] = dHdt


#------------------------------------------------------------------------------
# I/O
#------------------------------------------------------------------------------
def set_md_parameters(params, md):
    ''' set the parameters in an md instance

    Parameters
    ----------
    params : md_parameters object
        the simulation parameters
    md : md module instance
        the f2py md module for running simulation
    '''

    # set all the parameters
    n_bins = int(np.floor(params.n_particles**(1./3.)))
    bin_size = params.cell_size / n_bins

    md.parameters_mod.edgecellsize = bin_size
    md.parameters_mod.nxcell = n_bins
    md.parameters_mod.nycell = n_bins
    md.parameters_mod.nzcell = n_bins

    md.parameters_mod.timestep = params.timestep
    md.parameters_mod.novitotimestep = params.movie_rate

    md.parameters_mod.nspecies = params.n_species
    md.parameters_mod.nparticles = params.n_particles

    md.parameters_mod.screenlength = params.screen_length
    md.parameters_mod.cutoff = params.cutoff

    md.parameters_mod.extfield_x = params.ext_field[0]
    md.parameters_mod.extfield_y = params.ext_field[1]
    md.parameters_mod.extfield_z = params.ext_field[2]

    md.parameters_mod.ntimestepmts = params.n_mts_timesteps
    md.parameters_mod.cutoffmts = params.mts_cutoff

    # initialize OMP
    if md.parameters_mod.nprocomp == 0:
        md.parameters_mod.nprocomp = params.n_procs 
        md.initomp()

    # (deallocate if necessary and) allocate arrays
    if md.parameters_mod.initialized:
        logging.debug('cleaning up md module')
        md.cleanup()
    logging.debug('initializing md module')
    md.parameters_mod.thermostaton = True
    md.inisimulation()
    md.parameters_mod.thermostaton = False

    # set arrays with correct species information
    logging.debug('setting md species and equilibration parameters')
    sp_end = np.cumsum(params.particles)
    sp_start = sp_end - params.particles
    for sp in range(params.n_species):
        md.particles_mod.sp[sp_start[sp]:sp_end[sp]] = sp+1
        md.particles_mod.mass[sp_start[sp]:sp_end[sp]] = params.mass[sp]
        md.particles_mod.charge[sp_start[sp]:sp_end[sp]] = params.charge[sp]
        md.particles_mod.friction[sp_start[sp]:sp_end[sp]] = \
                params.friction[sp]
        md.particles_mod.ux[sp_start[sp]:sp_end[sp]] = \
                params.bulk_velocity[sp,0]
        md.particles_mod.uy[sp_start[sp]:sp_end[sp]] = \
                params.bulk_velocity[sp,1]
        md.particles_mod.uz[sp_start[sp]:sp_end[sp]] = \
                params.bulk_velocity[sp,2]
        md.particles_mod.avgke[sp_start[sp]:sp_end[sp],:] = \
                3. / 2. * params.stress[sp,np.arange(3),np.arange(3)]


def expand_md_box(params, md, factor=[2, 2]):
    ''' make the md box bigger by copying the particles and velocities

    copies the particle images and velocities to fill the bigger box and also
    rewrites the md parameters with the new settings

    Parameters
    ----------
    params : md_parameters object
        the simulation parameters
    md : md module
        the md simulation
    factor : 2-tuple, first entry must be integer (or will be rounded down)
        factor to enlarge box by, multiple for the screening length
    '''

    # get phasespace to copy
    pos, vel = get_md_phasespace(md)

    # set up
    factor[0] = int(factor[0])
    cell = params.cell_size
    old_particles = params.particles.copy()
    new_counter = 0
    old_counter = 0

    # change parameters to match new situation
    params.cell_size *= factor[0]
    params.particles *= factor[0]**3
    params.n_particles *= factor[0]**3
    params.cutoff *= factor[1]
    pos_new = np.empty((params.n_particles, 3))
    vel_new = np.empty((params.n_particles, 3))

    # perform copy and transformation on particles
    for sp in range(params.n_species):
        if sp > 0:
            old_counter += old_particles[sp-1]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    delta = np.array([i*cell, j*cell, k*cell])
                    pos_new[new_counter:new_counter+old_particles[sp],:] = \
                            (0.5 * cell +
                             pos[old_counter:old_counter+old_particles[sp],:] +
                             delta[np.newaxis,:])
                    vel_new[new_counter:new_counter+old_particles[sp],:] = \
                            vel[old_counter:old_counter+old_particles[sp],:]
                    new_counter += old_particles[sp]
    pos_new -= 0.5 * params.cell_size

    # reset md_simulation properties
    set_md_parameters(params, md)
    set_md_phasespace(pos_new, vel_new, md)


def set_md_phasespace(pos, vel, md):
    ''' set the phasespace in an md instance

    Parameters
    ----------
    pos : n_particles x 3 numpy array
        position of each particle
    vel : n_particles x 3 numpy array
        velocity of each particle
    md : md module instance
        the f2py md module for running simulation
    '''

    # set arrays with correct phasespace information
    logging.debug('setting md phase space')
    logging.debug(str(pos.shape[0]) + ' particles')
    md.particles_mod.rx[:] = pos[:,0]
    md.particles_mod.ry[:] = pos[:,1]
    md.particles_mod.rz[:] = pos[:,2]
    md.particles_mod.vx[:] = vel[:,0]
    md.particles_mod.vy[:] = vel[:,1]
    md.particles_mod.vz[:] = vel[:,2]


def get_md_phasespace(md):
    ''' get the phasespace from the md module
    
    Parameters
    ----------
    md : md module instance
        the f2py md module for running simulation

    Returns
    -------
    pos : n_particles x 3 numpy array of floats
        the x, y, z coordinates of each particle
    vel : n_particles long numpy array of floats
        the x, y, z velocities of each particle
    '''

    # allocate
    pos = np.empty((md.parameters_mod.nparticles, 3))
    vel = np.empty((md.parameters_mod.nparticles, 3))

    # read in positions and velocities
    pos[:,0] = md.particles_mod.rx[:]
    pos[:,1] = md.particles_mod.ry[:]
    pos[:,2] = md.particles_mod.rz[:]
    vel[:,0] = md.particles_mod.vx[:]
    vel[:,1] = md.particles_mod.vy[:]
    vel[:,2] = md.particles_mod.vz[:]

    return pos, vel


def read_md_phasespace(path='./out_md'):
    ''' read the MD phase space from the end of a simulation to get particle
    positions and velocities
    
    Parameters
    ----------
    path : string
        path to folder where the output file is

    Returns
    -------
    pos : n_particles x 3 numpy array of floats
        the x, y, z coordinates of each particle
    vel : n_particles long numpy array of floats
        the x, y, z velocities of each particle
    '''

    logging.debug('reading md phase space from file')
    if not path.endswith('/'):
        path += '/'
    data = np.fromfile(path + 'out_MD_phaseSpace.dat', dtype=float)
    data = data.reshape(data.shape[0] / 6, 6)
    pos = data[:,:3]
    vel = data[:,3:]
    return pos, vel


def get_md_forces(params, md):
    ''' read the forces on each particle due to each species from the md
    simulation module

    Parameters
    ----------
    params : md_parameters object
        the simulation parameters
    md : md module
        the md simulation

    Returns
    -------
    forces : n_particles x 3*n_species numpy array
    '''
 
    forces = np.empty((params.n_particles, 3*params.n_species))

    # loop over species for forces
    for sp in range(params.n_species):
        forces[:,3*sp+0] = md.particles_mod.forcefromspecies[0,sp,:]
        forces[:,3*sp+1] = md.particles_mod.forcefromspecies[1,sp,:]
        forces[:,3*sp+2] = md.particles_mod.forcefromspecies[2,sp,:]

    return forces


def get_md_conservation(md):
    ''' read the conservation data (total, potential, and kinetic energies)
    from the md simulation

    Parameters
    ----------
    md : md module
        the md simulation

    Returns
    -------
    energies : array of 3 floats
        the total, potential, and kinetic energies, in that order
    '''

    energies = np.array([md.particles_mod.totale, md.particles_mod.potentiale,
                         md.particles_mod.kinetice])

    return energies

def get_md_kinetic_energy(params, md):
    ''' get the kinetic energy for each species from the md simulation

    Parameters
    ----------
    params : md_parameters object
        the simulation parameters
    md : md module
        the md simulation

    Returns
    -------
    kinetic : n_species array of floats
        the kinetic energy per particle
    '''

    # setup
    sp_end = np.cumsum(params.particles)
    sp_start = sp_end - params.particles
    kinetic = np.empty((params.n_species,))

    # loop over species to get kinetic energy
    for sp in range(params.n_species):
        kinetic[sp] = (0.5 * params.mass[sp] *
                       np.sum(md.particles_mod.vx[sp_start[sp]:sp_end[sp]]**2 +
                              md.particles_mod.vy[sp_start[sp]:sp_end[sp]]**2 +
                              md.particles_mod.vz[sp_start[sp]:sp_end[sp]]**2) /
                       params.particles[sp])

    return kinetic


#------------------------------------------------------------------------------
# Generating positions and velocities
#------------------------------------------------------------------------------

def generate_md_phasespace(params):
    '''generate the MD phase space with initial particles and velocities
    given the simulation parameters

    Parameters
    ----------
    params : md_parameters object
        all the parameters from the simulation

    Returns
    -------
    species : n_particles x 1 numpy array of ints
        the species of each particle
    pos : n_particles x 3 numpy array of floats
        the x, y, z coordinates of each particle
    vel : n_particles long numpy array of floats
        the x, y, z velocities of each particle
    '''

    logging.debug('generating md positions and velocities with random sample')
    # generate positions and velocities
    bulk_velocity = np.array(params.bulk_velocity)
    species = np.vstack([(i+1)*np.ones((n, 1))
                         for i, n in enumerate(params.particles)])
    pos = uniform(low=-params.cell_size/2., high=params.cell_size/2.,
                  size=(params.n_particles, 3))
    vel = np.vstack([multivariate_normal(mean=bulk_velocity[i,:],
                                         cov=(params.stress[i] /
                                              params.mass[i] * np.eye(3)),
                                         size=n)
                     for i, n in enumerate(params.particles)])

    return species, pos, vel


def velocity_resample(distribution, params, atol=[1e-2, 1.5e-2, 2e-2],
                      rtol=[1e-2, 1e-2, 2e-2, 2e-2], debug=False):
    ''' resample the velocities given the distribution

    Parameters
    ----------
    distribution : list of distribution objects
        distributions for each species, must be indexable and generate samples
    pos : 3 x n_particles array-like of floats
        positions of each particle
    params : md_parameters object
        all the parameters from the simulation
    atols : 4 element array-like of floats
        absolute error tolerance for each moment
    rtols : 3 element array-like of floats
        relative error tolerance for moments 1 through 3

    Returns
    -------
    vel : 3 x n_particles array-like of floats
        velocities of each particle

    Draws samples from the distribution until the velocities match the moments
    sufficiently well. This is defined in the relative error sense as:
        (expected - observed) / expected < rtol OR (expected - observed) < atol
    We check absolute tolerance for the momentum, off-diagonal stresses, and
    heat transfer terms (where k is the moment number) by:
        abs((expected - observed) / (m * v_th**k))  < atol
    '''

    logging.debug('resampling the MD velocities')
    vel = np.empty((params.n_particles, 3))
    thermal_speed = np.sqrt(
        params.stress[:,np.arange(3),np.arange(3)].mean(axis=-1) / params.mass)

    # draw velocity samples for each species until get acceptable distribution
    counter = 0
    for s, n in enumerate(params.particles):
        trials = 0
        bad_sample = 1
        d_momentum = distribution[s].momentum
        d_stress = distribution[s].stress
        d_heat = distribution[s].heat
        d_m4 = distribution[s].m4
        while bad_sample:
            if not trials % 100:
                logging.debug('Species %d, attempt %d' % (s, trials))
            trials += 1
            bad_sample = 0
            tmp_vel = distribution[s].rvs(n)
            p_momentum, p_stress, p_kinetic_energy, p_heat, p_m4 = \
                    particle_moments(tmp_vel, params.mass[s])

            # check moments
            r_momentum_err = abs((d_momentum - p_momentum) / d_momentum)
            a_momentum_err = (abs((d_momentum - p_momentum) /
                              (params.mass[s] * thermal_speed[s])))
            r_stress_err = abs((d_stress - p_stress) / d_stress)
            a_stress_err = (abs((d_stress - p_stress) /
                              (params.mass[s] * thermal_speed[s]**2)))
            r_heat_err = abs((d_heat - p_heat) / d_heat)
            a_heat_err = (abs((d_heat - p_heat) /
                              (params.mass[s] * thermal_speed[s]**3)))
            r_m4_err = abs((d_m4 - p_m4) / d_m4)
            if not np.logical_or(r_momentum_err < rtol[0],
                                 a_momentum_err < atol[0]).all():
                bad_sample += 1
#               print('r_momentum_err ', r_momentum_err)
#               print('a_momentum_err ', a_momentum_err)
#               print('rtol: %f, atol: %f' % (rtol[0], atol[0]))
            if not np.logical_or(r_stress_err < rtol[1],
                                 a_stress_err < atol[1]).all():
                bad_sample += 10
#               print('r_stress_err ', r_stress_err)
#               print('a_stress_err ', a_stress_err)
#               print('rtol: %f, atol: %f' % (rtol[1], atol[1]))
            if not np.logical_or(
                    (abs((d_heat - p_heat) / d_heat) < rtol[2]),
                    (abs((d_heat - p_heat) /
                         (params.mass[s] * thermal_speed[s]**3))
                     < atol[2])).all():
                bad_sample += 100
#               print('r_heat_err ', r_heat_err)
#               print('a_heat_err ', a_heat_err)
#               print('rtol: %f, atol: %f' % (rtol[2], atol[2]))
            if not abs(r_m4_err < rtol[3]):
                bad_sample += 1000
#               print('r_m4_err ', r_m4_err)
#               print('rtol: %f' % (rtol[3]))
#           print('bad sample %d' % (bad_sample))

        logging.info('Species %d took %d trials.' % (s, trials))
        vel[counter:counter+n,:] = tmp_vel
        counter += n
    return vel


#------------------------------------------------------------------------------
# Running simulations
#------------------------------------------------------------------------------

def equilibrate_md(params, md, print_rate=10, save_rate=1000):
    ''' run the md equilibration phase for the desired time, assuming that the
    md has been initialized already

    Parameters
    ----------
    params : md_parameters object
        parameters of the simulation
    md : md module
        md simulation object
    print_rate : int
        how often to print the timestep
    save_rate : int
        how often to save the phasespace
        0 -> never

    Returns
    -------
    energy : n_timesteps+1 x 3 array of floats
        the total, potential, and kinetic energies throughout equilibration
    kinetic_energy : n_timesteps+1 x n_species array of floats
        the kinetic energy of each species throughout equilibration
    '''

    # setup
    n_steps = int(params.equilibration_time / (md.parameters_mod.timestep *
                                               params.plasma_frequency.min()))
    energy = np.empty((n_steps+1, 3))
    kinetic_energy = np.empty((n_steps+1, params.n_species))


    # initialize simulation
    md.parameters_mod.thermostaton = True
    md.forces()
    md.conservation()
    energy[0,:] = get_md_conservation(md)
    kinetic_energy[0,:] = get_md_kinetic_energy(params, md)

    # loop over time
    logging.info('starting equilibration')
    for step in range(1, n_steps+1):
        if step % print_rate == 0:
            logging.info('equilibration step %d of %d' % (step, n_steps))
        md.evl()
        md.conservation()
        energy[step,:] = get_md_conservation(md)
        kinetic_energy[step,:] = get_md_kinetic_energy(params, md)

    return energy, kinetic_energy


def simulate_md(params, distribution, md, print_rate=10, resample=True,
                atol=[1e-2, 1.5e-2, 2e-2], rtol=[1e-2, 1.5e-2, 2e-2, 2e-2],
                refresh_rate=0, save_rate=1000, resume=False, last_step=0,
                current_sim=0):
    ''' run the specified number of md simulations for the desired number of
    time steps and collect data on energy, dHdt, and moments

    Parameters
    ----------
    params : md_parameters object
        parameters of the simulation
    distribution : array of distribution objects
        the distributions to sample from
    md : md module
        md simulation object
    print_rate : int
        how often to print the timestep
    resample : boolean
        whether to do velocity resampling before each trial
    atol : array of 3 floats
        absolute tolerance for resampling
    rtol : array of 3 floats
        relative tolerance for resampling
    refresh_rate : int
        how often to refresh the distribution based on average temperature
        0 -> never
    save_rate : int
        how often to save the phasespace and data for restart
        0 -> never
    resume : boolean
        whether resuming from a previous simulation
    last_step : int
        step to resume from
        0 -> not resuming
    current_sim : int
        current simulation (for adding sims)

    Returns
    -------
    energy : n_timesteps+1 x 3 array of floats
        the total, potential, and kinetic energies throughout equilibration
    data : md_data object
        all the moments and dHdt from throughout the simulation
    '''


    # setup
    energy = np.empty((params.n_sims, params.n_timesteps+1, 3))
    data = md_data(params)
    md.parameters_mod.thermostaton = False
    pos0, vel0 = get_md_phasespace(md)
    prev_r = np.empty((3, params.n_particles))
    prev_v = np.empty((3, params.n_particles))
    prev_a = np.empty((3, params.n_particles))
    if refresh_rate > 0:
        refresh_rate = int(refresh_rate)
        distribution_log = np.empty((params.n_sims,
                                     params.n_timesteps/refresh_rate+1,
                                     params.n_species), dtype=object)
        distribution_log[:,0,:] = \
                np.array(distribution)[np.newaxis,np.newaxis,:]
        log_counter = 1

    # loop over simulations
    for sim in range(current_sim, params.n_sims):
#        logging.info('---------------------------------------------------')
#        logging.info('SIMULATION %d of %d' % (sim+1, params.n_sims))
#        logging.info('---------------------------------------------------')

        # velocity resample if needed
        if resample and not resume:
            vel = velocity_resample(distribution, params, atol, rtol)
            set_md_phasespace(pos0, vel, md)

        # intialize simulation
        md.forces()

        md.conservation()
        md.movie()
        md.naughtypair()
        if current_sim > 0:
            logging.info('reloading simulation data')
            energy[:-1] = np.load('energy.npy')
            data.dHdt[:-1] = np.load('data.dHdt.npy')
            data.momentum[:-1] = np.load('data.momentum.npy')
            data.stress[:-1] = np.load('data.stress.npy')
            data.kinetic_energy[:-1] = np.load('data.kinetic_energy.npy')
            data.heat[:-1] = np.load('data.heat.npy')
            data.m4[:-1] = np.load('data.m4.npy')

        if not resume:
            energy[sim,0,:] = get_md_conservation(md)
            data.get_md_data(sim, 0, params, md, distribution)
        else:
            logging.info("reinitializing previous simulation data")
            energy[sim,:last_step+1,:] = \
                    np.load('energy.npy')[sim,:last_step+1,:]
            data.dHdt[sim,:last_step+1,:,:] = \
                    np.load('data.dHdt.npy')[sim,:last_step+1,:,:]
            data.momentum[sim,:last_step+1,:,:] = \
                    np.load('data.momentum.npy')[sim,:last_step+1,:,:]
            data.stress[sim,:last_step+1,:,:,:] = \
                    np.load('data.stress.npy')[sim,:last_step+1,:,:,:]
            data.kinetic_energy[sim,:last_step+1,:] = \
                    np.load('data.kinetic_energy.npy')[sim,:last_step+1,:]
            data.heat[sim,:last_step+1,:,:] = \
                    np.load('data.heat.npy')[sim,:last_step+1,:,:]
            data.m4[sim,:last_step+1,:] = \
                    np.load('data.m4.npy')[sim,:last_step+1,:]

        # loop over time
        for step in range(last_step+1, params.n_timesteps+1):
            if step % print_rate == 0:
                logging.info('timestep %d of %d for simulation %d of %d' %
                      (step, params.n_timesteps, sim+1, params.n_sims))
                logging.info('total energy jump so far: %f',
                             (energy[sim,step-1,0] - energy[sim,0,0]) / 
                             energy[sim,0,0])
            prev_r[0,:] = md.particles_mod.rx[:]
            prev_r[1,:] = md.particles_mod.ry[:]
            prev_r[2,:] = md.particles_mod.rz[:]
            prev_v[0,:] = md.particles_mod.vx[:]
            prev_v[1,:] = md.particles_mod.vy[:]
            prev_v[2,:] = md.particles_mod.vz[:]
            prev_a[0,:] = md.particles_mod.ax[:]
            prev_a[1,:] = md.particles_mod.ay[:]
            prev_a[2,:] = md.particles_mod.az[:]

            md.evl()
            if md.parameters_mod.outofbounds:
                raise ValueError('particle flew out of the domain')

            md.conservation()
            md.naughtypair()
            energy[sim,step,:] = get_md_conservation(md)

            # MTS
            delta = (energy[sim,step,0]-energy[sim,step-1,0]) / energy[sim,0,0]
            if abs(delta) > params.mts_threshold:
                logging.debug('taking MTS step at step %d' % step)
                logging.debug('energy jump is %f' % delta)
                totjump = ((energy[sim,step-1,0] - energy[sim,0,0]) /
                           energy[sim,0,0])
                logging.debug('total energy change before this step was %f' %
                              totjump)
                logging.debug('max acceleration is: %f' %
                              md.particles_mod.maxaij)
                md.particles_mod.rx = prev_r[0,:]
                md.particles_mod.ry = prev_r[1,:]
                md.particles_mod.rz = prev_r[2,:]
                md.particles_mod.vx = prev_v[0,:]
                md.particles_mod.vy = prev_v[1,:]
                md.particles_mod.vz = prev_v[2,:]
                md.particles_mod.ax = prev_a[0,:]
                md.particles_mod.ay = prev_a[1,:]
                md.particles_mod.az = prev_a[2,:]

                md.evlmts()
                if md.parameters_mod.outofbounds:
                    raise ValueError('particle flew out of the domain')

                logging.debug('Number of MTS particles: %d' %
                      np.sum(md.particles_mod.ismts))
                md.conservation()
                energy[sim,step,:] = get_md_conservation(md)

            if step % params.movie_rate == 0:
                md.movie()
            data.get_md_data(sim, step, params, md, distribution)

            # save data every save_rate timesteps
            if (save_rate > 0 and step % save_rate == 0) or step==params.n_timesteps:
                logging.debug('saving simulation data at time step %d' % step)
                pos, vel = get_md_phasespace(md)

                # write output to files
                np.save('energy', energy)
                np.save('data.momentum', data.momentum)
                np.save('data.stress', data.stress)
                np.save('data.kinetic_energy', data.kinetic_energy)
                np.save('data.heat', data.heat)
                np.save('data.m4', data.m4)
                np.save('data.dHdt', data.dHdt)
                np.save('data.mass', data.mass)
                np.save('data.time', data.time)
                np.save('end_pos', pos)
                np.save('end_vel', vel)

            # update distributions if needed
            if refresh_rate > 0 and step % refresh_rate == 0:
                logging.debug('refreshing the distributions')
                average_ke = data.kinetic_energy[sim,step-refresh_rate:step,:]
                average_ke = np.squeeze(average_ke).mean(axis=0)
                for sp in range(params.n_species):
                    vx = distribution[sp]._x
                    vy = distribution[sp]._y
                    vz = distribution[sp]._z
                    f = distributions.discrete_maxwellian3D(
                        vx, vy, vz, params.mass[sp], params.density[sp],
                        np.zeros((3,)), average_ke[sp])
                    distribution[sp] = distributions.linear_interpolated_rv_3D(
                        vx, vy, vz, f, params.mass[sp])
                distribution_log[sim,log_counter,:] = distribution
                log_counter += 1

        # do another equilibration bewteen simulations
        if sim < params.n_sims-1:
            md.closefiles()
            md.openfiles()
            logging.debug('equilibrating between MD simulations')
            set_md_phasespace(pos0, vel0, md)
            equilibrate_md(params, md, print_rate=200, save_rate=save_rate)
            md.parameters_mod.thermostaton = False
            md.closefiles()
            md.openfiles()


    if refresh_rate > 0:
        return energy, data, distribution_log
    else:
        return energy, data, pos0, vel0


def setup_md_workspace():
    ''' create the folders for the input/output files and create input files
    with default values as a placeholder
    '''

    logging.debug('creating necessary md folders')
    # set up folders
    contents = os.listdir('.')
    if 'out_md' not in contents:
        os.mkdir('./out_md')
