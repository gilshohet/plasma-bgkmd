''' set BGK parameters, run the simulation, and read the data files

Given the simulation conditions, initialize and run BGK simulations, and read
the output from the simulation.
'''

import distributions
import units
import numpy as np
import os
import subprocess

#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

class bgk_parameters(object):
    ''' class with all the parameters for a BGK simulation

    EVERYTHING IN CGS UNITS (except distribution, which is converted)

    case : string
        name of the test case
    n_dims : int, 0 or 1
        number of space dimensions, 0 for 0D-3V, 1 for 1D-3V
    length : float
        length of domain (in cm)
    n_cells : int
        number of cells in physical space
    n_vel : int
        number of velocity points in each direction (total is Nv**3)
    timestep : float
        time step (in seconds)
    current_time : float
        current time 
    run_time : float
        length of time to simulate (in seconds)
    order : int, 1 or 2
        order of accuracy, 1 for first-order, 2 for second-order
    implicit : boolean
        whether to do implicit time stepping
    data_rate : int
        number of time steps between writing to output files
    n_species : int
        number of species
    mass : array-like float
        mass of each species (in grams)
    charge : array-like float
        charge number (Z) of each species (charge = Z * e)
    distribution : list of distribution objects
        the discrete distribution associated with each species
    taus : n_species x n_species numpy array
        the interspecies and intraspecies taus
    run_to_completion : boolean
        whether to use the 2-norm-based stopping condition
    bgk_path : string
        path to the BGK program root folder
    '''

    def __init__ (self, case='test', n_dims=0, length=0, n_cells=1, n_vel=100,
                  timestep=1.0e-14, current_time=0.0, run_time=2.0e-11, order=1,
                  implicit=False, data_rate=1, n_species=1, charge=[1.0],
                  distribution=None, taus=None, run_to_completion=True,
                  rhs_tol=1.0, bgk_path='.'):
        if distribution is None:
            raise ValueError('Need to specify the distributions at a minimum.')
        self.case = case
        self.n_dims = n_dims
        self.length = length
        self.n_cells = n_cells
        self.n_vel = n_vel
        self.timestep = timestep
        self.current_time = current_time
        self.run_time = run_time
        self.order = order
        self.implicit = implicit
        self.data_rate = data_rate
        self.n_species = n_species
        self.charge = np.array(charge)
        self.distribution = distribution
        self.mass = np.empty(n_species)
        self.taus = taus
        self.rhs_tol = rhs_tol
        self.run_to_completion = run_to_completion
        for sp in range(n_species):
            self.mass[sp] = (distribution[0,sp].mass / units.g)
        if not bgk_path.endswith('/'):
            bgk_path += '/'
        self.bgk_path = bgk_path


class bgk_data(object):
    ''' object for storing output of given BGK simulation to make plotting
    time dependent quantities easier

    x_velocity : n_saves x n_species
        the x bulk_velocity for each species at each time
    x_momentum : n_saves x n_species
        the x momentum per particle for each species at each time
    kinetic_energy : n_saves x n_species numpy array
        the kinetic energy per particle for each species at each time
    time : n_saves numpy array
        physical simulation time at each save point
    '''

    def __init__(self, params):
        ''' load the data and generate the data object, converting everything
        to atomic units

        Parameters
        ----------
        params : bgk_parameters object
            parameters of the simulation
        '''

        # initialize
        n_saves = int(params.run_time / params.timestep / 
                      params.data_rate + 1)
        self.time = params.timestep * params.data_rate * np.arange(n_saves)
        self.x_velocity = np.empty((n_saves, params.n_species, 3))
        self.kinetic_energy = np.empty((n_saves, params.n_species))
        self.x_momentum = np.empty((n_saves, params.n_species))

        # loop over species and load data
        for sp in params.n_species:
            self.x_velocity[:,sp] = np.loadtxt(params.bgk_path + 'input/' +
                                               case + '_velo' + str(sp))
            self.kinetic_energy[:,sp] = np.loadtxt(params.bgk_path + 'input/' +
                                             case + '_temp' + str(sp)) * 3./2.
            self.x_momentum[:,sp] = self.x_velocity[:,sp] * params.mass[sp]

        # convert units
        self.x_velocity *= units.s / units.cm
        self.x_momentum *= units.s / units.cm / units.g
        self.kinetic_energy /= units.eV

#------------------------------------------------------------------------------
# File I/O
#------------------------------------------------------------------------------

def write_bgk_parameters(params):
    '''write the BGK input file given the MD parameters

    Parameters
    ----------
    params : bgk_parameters object
        all the parameters for the simulation
    '''

    path = params.bgk_path + 'input/' + params.case
    with open(path, 'w') as f:
        # user defined parameters
        f.write('dims\n')
        f.write('%d\n' % (params.n_dims))
        f.write('\n')
        f.write('Lx\n')
        f.write('%.8E\n' % (params.length))
        f.write('\n')
        f.write('Nx\n')
        f.write('%d\n' % (params.n_cells))
        f.write('\n')
        f.write('Nv\n')
        f.write('%d\n' % (params.n_vel))
        f.write('\n')
        f.write('Time_step\n')
        f.write('%.8E\n' % (params.timestep))
        f.write('\n')
        f.write('Final_time\n')
        f.write('%.8E\n' % (params.current_time + params.run_time))
        f.write('\n')
        f.write('Space_order\n')
        f.write('%d\n' % (params.order))
        f.write('\n')
        f.write('Imp_exp\n')
        f.write('%d\n' % (int(params.implicit)))
        f.write('\n')
        f.write('Data_writing_frequency\n')
        f.write('%d\n' % (params.data_rate))
        f.write('\n')
        f.write('nspec\n')
        f.write('%d\n' % (params.n_species))
        f.write('\n')
        f.write('mass\n')
        for mass in params.mass:
            f.write('%.8E\n' % (mass))
        f.write('\n')
        f.write('Z\n')
        for charge in params.charge:
            f.write('%.8E\n' % (charge))
        f.write('\n')
        f.write('RHS_tol\n')
        f.write('%.8E\n' % (params.rhs_tol))
        f.write('\n')

        # other stuff for initializing the simulation
        f.write('discret\n')
        f.write('0\n')
        f.write('ecouple\n')
        f.write('0\n')
        f.write('\n')
        f.write('Ion_coll_type\n')
        f.write('0\n')
        f.write('\n')
        f.write('Ion_coll_flavor\n')
        f.write('1\n')
        f.write('\n')
        f.write('Coulomb_type\n')
        f.write('0\n')
        f.write('\n')

        # also write the distribution details, to avoid segfault?
        f.write('n\n')
        for sp in range(params.n_species):
            density = params.distribution[0,sp].density * units.cc
            f.write('%.8E\n' % (density))
        f.write('\n')
        f.write('v\n')
        for sp in range(params.n_species):
            vel = (params.distribution[0,sp].momentum / params.mass[sp] * 
                   units.s / units.cm / units.g)
            f.write('%.8E\n' % (vel[0]))
        f.write('\n')
        f.write('T\n')
        for sp in range(params.n_species):
            T = (params.distribution[0,sp].kinetic_energy * 2./3. / units.eV)
            f.write('%.8E\n' % (T))
        f.write('Stop')


def read_distributions0D(params):
    ''' read the distributions from the bgk simulation for use in the tau
    calculation, and convert units to atomic

    Parameters
    ----------
    params : bgk_parameters object
        all the parameters for the simulation

    Returns
    -------
    distribution : array of distribution objects
        distribution for each species
    time : float
        current simulation time (in seconds)
    '''

    distribution = []
    path = params.bgk_path + 'Data/' + params.case + '_gridinfo.dat'
    with open(path, 'r') as f:
        for sp in range(params.n_species):
            line = f.readline().split()
            Nv = int(line[1])
            v_max = float(line[2]) * units.cm / units.s
            vx = vy = vz = np.linspace(-v_max, v_max, Nv)
            data = np.fromfile(params.bgk_path + 'Data/' + params.case +
                               '_spec' + str(sp) + '.dat')
            dist = data.reshape((Nv, Nv, Nv)) * units.s**3 / units.cm**6
            mass = params.mass[sp] * units.g
            dist = distributions.linear_interpolated_rv_3D(vx, vy, vz, dist,
                                                           mass)
            distribution.append(dist)
        time = float(f.readline())

    distribution = np.array(distribution, dtype=object)
    return distribution.reshape((params.n_cells, params.n_species)), time


def write_distributions0D(params):
    ''' write the distributions to a binary file to be read by the BGK code

    Parameters
    ----------
    params : bgk_parameters object
        all the parameters for the simulation
    '''

    path = params.bgk_path + 'Data/' + params.case + '_gridinfo.dat'
    # first write the discretization
    with open(path, 'w') as f:
        for sp in range(params.n_species):
            f.write('%d %d %.5E\n' % (sp, params.n_vel,
                    params.distribution[0,sp]._x[-1] * units.s / units.cm))
        f.write('%.8E' % params.current_time)
    # then write the distributions
    path = params.bgk_path + 'Data/' + params.case
    for sp in range(params.n_species):
        dist = (params.distribution[0,sp].distribution / 
                units.s**3 * units.cm**6)
        dist.tofile(path + '_spec' +  str(sp) + '.dat')


def write_taus(params):
    path = params.bgk_path + 'Data/' + params.case + '_tau.dat'
    with open(path, 'w') as f:
        for sp1 in range(params.n_species):
            for sp2 in range(params.n_species):
                f.write('%.8e\n' % (params.taus[0,sp1,sp2]))


#------------------------------------------------------------------------------
# Running simulations
#------------------------------------------------------------------------------

def run_bgk_simulation(params):
    ''' write distributions and input files and launch a BGK simulation

    Performs the following:
        1) write the parameters to the bgk input file with test case name
        2) write the distributions and velocity discretizations to Data
        3) write the taus, if using user-defined taus
        4) run the bgk simulation

    Parameters
    ----------
    params : bgk_parameters object
        all the parameters for the simulation
    '''

    # write parameters
    write_bgk_parameters(params)
    
    # write distributions
    if params.n_dims is 0:
        write_distributions0D(params)
    else:
        raise ValueError('1D not currently supported')

    if params.taus is not None:
        write_taus(params)
        tau_flag = 1
    else:
        tau_flag = 0

    if params.run_to_completion:
        restart_flag = 2
    else:
        restart_flag = 4
    
    # go to BGK directory
    start_path = os.getcwd()
    os.chdir(params.bgk_path)
    
    # run the simulation
    try:
        subprocess.check_call(['exec/MultiBGK_ ' +  params.case + ' ' + 
                               str(restart_flag) + ' ' +
                               str(tau_flag)], shell=True)
    except:
        os.chdir(start_path)
        raise
    os.chdir(start_path)

