''' tools for running an hmm using the MD and BGk codes

All of the file i/o and control structures necessary to run an HMM simulation
given the simulation parameters.

Currently only 0-D will work.

Will at some point be set up to save current state on errors and keyboard
interrupt, and possibly resume. Maybe. This may never actually happen.
'''

import md
import md_io
import bgk_io
import tau_utils
import distributions
import moments
import units
import numpy as np
import scipy.stats as stats
import os
import logging


#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

class simulation(object):
    ''' class to store all the simulation parameters and data needed to run
    the bgk and md simulations and run the actual simulations

    General parameter inputs
    ------------------------
    testcase : string
        name for the test case being run
    only_md : boolean
        flag whether to just run an MD simulation instead of full HMM, don't
        need bgk parameters except for n_vel if this is on. tau_update_rate is
        optional and will set the distribution refresh rate in MD.

    Species parameter inputs
    ------------------------
    n_species : int
        number of species
    mass : n_species float array
        mass of each species (in grams)
    charge : n_species float array
        charge (Z) of each species
    initial_density : n_cells x n_species float array
        the initial density of each species in each cell (per cc)
    initial_temperature : n_cells x n_species float array
        the initial temperature of each species in each cell (in eV)
    initial_velocity : n_cells x n_species float array
        the initial x-velocity of each species in each cell (in cm/s)

    BGK simulation parameter inputs
    -------------------------------
    n_dim : int (0 or 1)
        number of dimensions
    n_vel : int
        number of velocity gridpoints in each direction
    order : int (1 or 2)
        whether to be first or second order
    implicit : boolean
        whether to use implicit time stepping
    timestep_bgk : float
        timestep to use in bgk (in seconds)
    final_time : float
        end time of the simulation (in seconds)
    n_cells_bgk : int
        number of bgk cells (if 1D, otherwise this is ignored)
    cell_length_bgk : float
        length of bgk simulation cell in cm (if 1D, otherwise this is ignored)
    tau_update_rate : int
        how many bgk steps to take between updates to the taus
    data_rate_bgk : int
        how often to write data to file in the bgk simulations
    bgk_path : string
        path to the main BGK folder

    MD simulation parameter inputs
    ------------------------------
    timestep_md : float
        timestep to take in md as proportion of 1 / plasma_frequency
    n_timesteps_md : int
        number of timesteps to take in each md simulation
    max_timesteps_md : int
        max number of timesteps to take in each md simulation if too noisy
    friction : array of 2 floats
        equilibration power as function of plasma_frequency, for the strong
        and weak thermostat phases
    equilibration_time : array of 2 floats
        time to euqilibrate with strong and weak thermostat in terms of the
        slowest plasma period
    small_box_equilibration : float
        how long to equilibrate on a box half the size first in terms of
        slowest plasma period, will not do this stage if 0
    n_simulations_md : int
        number of md simulations to run when for computation of dH/dt
    cell_length_md : float
        length of side of md cell as proportion of screening length
    cutoff_md : float
        cutoff radius for computing forces as proportion of screening length
    mts_cutoff_md : float
        cutoff radius for MTS as proportion of screening length
    mts_threshold_md : float
        allowable relative change in total energy before doing MTS
    mts_timesteps_md : int
        number of MTS timesteps in one regular timestep
    movie_rate : int
        number of timesteps between saving data in ovito movie format
    md_save_rate : int
        how often to save md phasespace
    md_resample : boolean
        whether to do the velocity resample after equilibration
    md_nprocs : int
        limit how many processors to use in MD (default all)
    md_resume : bool
        whether resuming from a previous simulation
    md_last_step : int 
        last step taking if resuming, 0 else

    Simulation state parameters (not loaded from input file)
    --------------------------------------------------------
    time : float
        current simulation time (in seconds)
    distributions : list of distribution objects
        the current distribution of each species
    taus : n_species x n_species float array
        current relaxation times
    tau_hist : list of n_species x n_species arrays
        time history of the taus every time we compute them
    tau_error_hist : list of n_species x n_species arrays
        time history of the relative error in dHdt from the taus computed
    '''
    
    def __init__(self, infile):
        ''' initialize the hmm_data object with the input file

        Parameters
        ----------
        infile : string
            the path to the input file
        '''

        # load the inputs
        logging.info('Reading in HMM input file')
        self.process_input_file(infile)        

        # only 0-D for now
        if self.n_dim is 1:
            raise ValueError('only 0D is supported at the moment')

        # set up the workspace
        logging.info('Setting up workspace for HMM')
        self.setup()

        # initialize everything
        logging.info('Initializing simulation')
        self.initialize()

    
    def read_input_file(self, infile):
        ''' read the input file and convert to a dictionary
        
        Parameters
        ----------
        infile : string
            the path to the input file

        Returns
        -------
        parameters : dictionary
            all the parameters in the input file in dict format
        '''

        parameters = {}
        with open(infile, 'r') as f:
            for line in f:
                if not line.startswith('#') and ':' in line:
                    line = line.strip().replace(' ','')
                    while line.endswith(','):
                        line += next(f).strip().replace(' ','')
                    (key, val) = line.split(':')
                    parameters[key] = val

        return parameters

    def process_input_file(self, infile):
        ''' read the input file, check that all required parameters are
        present, and convert everything to the correct data types

        inputs
        ------
        infile : string
            the path to the input file
        '''

        # load the inputs
        param_dict = self.read_input_file(infile)

        # go through the parameters, convert to correct dtypes, and sanitize
        try:
            self.testcase = param_dict['testcase']
            logging.debug('testcase: ' + self.testcase)
        except KeyError:
            err = 'missing parameter \'testcase\''
            raise KeyError(err)

        try:
            self.only_md = int(bool(param_dict['only_md']
                                    not in ['F', 'f', 'False', 'false', '0']))
            logging.debug('only doing MD' if self.only_md else 'doing full HMM')
        except KeyError:
            self.only_md = 0
            logging.debug('doing full HMM')

        try:
            self.n_species = int(param_dict['n_species'])
            logging.debug('n_species: %d' % self.n_species)
        except KeyError:
            err = 'missing parameter \'n_species\''
            raise KeyError(err)

        try:
            self.mass = np.fromstring(param_dict['mass'], sep=',') * units.g
            logging.debug('mass: ' + np.array_str(self.mass))
            if len(self.mass) is not self.n_species:
                raise ValueError
        except ValueError:
            err = '\'mass\' must be n_species comma-separated floats'
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'mass\''
            raise KeyError(err)

        try:
            self.charge = np.fromstring(param_dict['charge'], sep=',')
            logging.debug('charge: ' + np.array_str(self.charge))
            if len(self.charge) is not self.n_species:
                raise ValueError
        except ValueError:
            err = '\'charge\' must be n_species comma-separated floats'
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'charge\''
            raise KeyError(err)

        try:
            self.n_dim = int(param_dict['n_dim'])
            logging.debug('number of spatial dimensions: %d' % self.n_dim)
            if self.n_dim is not 0 and self.n_dim is not 1:
                raise ValueError
        except ValueError:
            err = 'number of dimensions must be either zero or one'
            raise ValueError(err)
        except KeyError:
            if self.only_md:
                self.n_dim = 0
            else:
                err = 'missing parameter \'n_dim\''
                raise KeyError(err)

        try:
            self.n_vel = int(param_dict['n_vel'])
            logging.debug('number of velocity gridpoints: %d' % self.n_vel)
        except KeyError:
            err = 'missing parameter \'n_vel\''
            raise KeyError(err)

        try:
            self.order = int(param_dict['order'])
            logging.debug('order of accuracy: %d' % self.order)
            if self.order is not 1 and self.order is not 2:
                raise ValueError
        except ValueError:
            err = 'order must be either one or two'
            raise ValueError(err)
        except KeyError:
            self.order = 1
            logging.debug('order not specified, defaulting to first-order')

        try:
            self.implicit = int(bool(param_dict['implicit']
                                     not in ['F', 'f', 'False', 'false', '0']))
            logging.debug(('implicit' if self.implicit else 'explicit') +
                          ' time step')
        except KeyError:
            self.implicit = 0
            logging.debug('explicit/implicit not specified, defaulting to' + 
                          ' explicit time step.')

        try:
            self.timestep_bgk = float(param_dict['timestep_bgk'])
            logging.debug('bgk time step: %f seconds' % self.timestep_bgk)
        except KeyError:
            err = 'missing parameter \'timestep_bgk\''
            if not self.only_md:
                raise KeyError(err)

        try:
            self.final_time = float(param_dict['final_time'])
            logging.debug('final simulation time: %f seconds' %
                          self.final_time)
        except KeyError:
            err = 'missing parameter \'final_time\''
            if not self.only_md:
                raise KeyError(err)
        
        if self.n_dim is 1:
            try:
                self.n_cells_bgk = int(param_dict['n_cells_bgk'])
                logging.debug('number of spatial cells: %d' % self.n_cells_bgk)
            except KeyError:
                err = 'missing parameter \'n_cells_bgk\''
                raise KeyError(err)
        else:
            self.n_cells_bgk = 1
            logging.debug('number of spatial cells: 1')

        if self.n_dim is 1:
            try:
                self.cell_length_bgk = float(param_dict['cell_length_bgk'])
                logging.debug('bgk cell length is: %f' % self.cell_length_bgk)
            except KeyError:
                err = 'missing parameter \'cell_length_bgk\''
                raise KeyError(err)
        else:
            self.cell_length_bgk = 0.
            logging.debug('zero dimensions, bgk cell length is meaningless')

        try:
            self.tau_update_rate = int(param_dict['tau_update_rate'])
            logging.debug('updating tau every %d steps' % self.tau_update_rate)
        except KeyError:
            err = 'missing parameter \'tau_update_rate\''
            if self.only_md:
                tau_update_rate = 0
            else:
                raise KeyError(err)

        try:
            self.data_rate_bgk = int(param_dict['data_rate_bgk'])
            logging.debug('outputting bgk data every %d steps' %
                          self.data_rate_bgk)
        except KeyError:
            self.data_rate_bgk = 1
            logging.debug('bgk data rate unspecified, defaulting to every step')

        try:
            self.bgk_path = param_dict['bgk_path']
            logging.debug('looking at path: ' + self.bgk_path +
                          ' for BGK executable and input/output files')
        except KeyError:
            err = 'missing parameter \'bgk_path\''
            if not self.only_md:
                raise KeyError(err)

        try:
            self.timestep_md = float(param_dict['timestep_md'])
            logging.debug('MD timestep: %f plasma periods' % self.timestep_md)
        except KeyError:
            err = 'missing parameter \'timestep_md\''
            raise KeyError(err)

        try:
            self.n_timesteps_md = int(param_dict['n_timesteps_md'])
            logging.debug('number of MD steps: %d' % self.n_timesteps_md)
        except KeyError:
            err = 'missing parameter \'n_timesteps_md\''
            raise KeyError(err)

        try:
            self.max_timesteps_md = int(param_dict['max_timesteps_md'])
            logging.debug('max number of MD steps: %d' % self.max_timesteps_md)
        except KeyError:
            self.max_timesteps_md = self.n_timesteps_md

        try:
            self.equilibration_time = \
                    np.fromstring(param_dict['equilibration_time'],
                                  sep=',')
            logging.debug('equilibration times: ' +
                          np.array_str(self.equilibration_time) +
                          ' plasma periods')
            if len(self.equilibration_time) is not 2:
                err = ('expecting two comma separated floats for' + 
                       ' equilibration time')
                raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'equilibration_time\''
            raise KeyError(err)

        try:
            self.small_box_equilibration =\
                    float(param_dict['small_box_equilibration'])
            logging.debug('equilibrating on small box for %f plasma periods' %
                          self.small_box_equilibration)
        except KeyError:
            self.small_box_equilibration = 0
            print('defaulting to equilibration on full particle set')

        try:
            self.friction = np.fromstring(param_dict['friction'], sep=',')
            logging.debug('using equilibration friction: ' +
                          np.array_str(self.friction))
            if len(self.friction) is not 2:
                raise ValueError
        except ValueError:
            err = ('expecting two comma separated floats for' + 
                    'equilibration friction')
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'friction\''
            raise KeyError(err)

        try:
            self.n_simulations_md = int(param_dict['n_simulations_md'])
            logging.debug('md simulations per run: %d' % self.n_simulations_md)
        except KeyError:
            err = 'missing parameter \'n_simulations_md\''
            raise KeyError(err)

        try:
            self.cell_length_md = float(param_dict['cell_length_md'])
            logging.debug('md cell length: %f screening lengths' %
                          self.cell_length_md)
        except KeyError:
            err = 'missing parameter \'cell_length_md\''
            raise KeyError(err)

        try:
            self.cutoff_md = float(param_dict['cutoff_md'])
            logging.debug('cutoff radius: %f screening lengths' %
                          self.cutoff_md)
        except KeyError:
            err = 'missing parameter \'cutoff_md\''
            raise KeyError(err)

        try:
            self.mts_cutoff_md = float(param_dict['mts_cutoff_md'])
            logging.debug('MTS cutoff: %f screening lengths' %
                          self.mts_cutoff_md)
        except KeyError:
            logging.debug('missing key mts_cutoff_md, defaulting to 0.0' + 
                  ' (only do MTS on the two particles)')
            self.mts_cutoff_md = 0.0

        try:
            self.mts_threshold_md = float(param_dict['mts_threshold_md'])
            logging.debug('MTS energy jump threshold: %f' %
                          self.mts_threshold_md)
        except KeyError:
            logging.debug('missing key mts_threshold_md, defaulting to 100 ' +
                          '(no mTS)')
            self.mts_threshold_md = 100.

        try:
            self.mts_timesteps_md = float(param_dict['mts_timesteps_md'])
            logging.debug('number of small timesteps for MTS: %d' %
                          self.mts_timesteps_md)
        except KeyError:
            loggging.debug('missing key \'mts_timesteps_md\', default is 100')
            self.mts_timesteps_md = 100

        try:
            self.movie_rate = int(param_dict['movie_rate'])
            logging.debug('writing movie every %d timesteps' % self.movie_rate)
        except KeyError:
            err = 'missing parameter \'movie_rate\''
            raise KeyError(err)

        try:
            self.md_save_rate = int(param_dict['md_save_rate'])
            logging.debug('writing phasespace every %d timesteps' % self.md_save_rate)
        except KeyError:
            logging.debug('missing key \'md_save_rate\', default is never')
            self.md_save_rate = 0

        try:
            self.md_resample = bool(param_dict['md_resample'])
        except KeyError:
            self.md_resample = True

        try:
            self.md_nprocs = int(param_dict['md_nprocs'])
        except KeyError:
            self.md_nprocs = 1000000

        try:
            self.md_resume = bool(param_dict['md_resume'])
        except KeyError:
            self.md_resume = False

        if self.md_resume:
            try:
                self.md_last_step = int(param_dict['md_last_step'])
                logging.info('Resuming at time step ' + str(self.md_last_step))
            except KeyError:
                err = "missing key \'md_last_step\' for resume"
                raise KeyError(err)
        else:
            self.md_last_step = 0

        try:
            initial_density = (np.fromstring(param_dict['initial_density'], 
                                             sep=',') / units.cm**3) 
            if len(initial_density) != self.n_species * self.n_cells_bgk:
                raise ValueError
            self.initial_density = initial_density.reshape((self.n_cells_bgk,
                                                            self.n_species))
            logging.debug('initial density: ' +
                          np.array_str(self.initial_density))
        except ValueError:
            err = ('\'density\' must be n_species x n_cells_bgk' +
                   ' comma-separated floats')
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'initial_density\''
            raise KeyError(err)
        
        try:
            initial_temperature = \
                    (np.fromstring(param_dict['initial_temperature'], sep=',')
                     * units.eV)
            if (len(initial_temperature) is not 
                    self.n_species * self.n_cells_bgk):
                raise ValueError
            self.initial_temperature = initial_temperature.reshape(
                    (self.n_cells_bgk, self.n_species))
            logging.debug('initial temperature: ' +
                          np.array_str(self.initial_temperature))
        except ValueError:
            err = ('\'temperature\' must be n_species x n_cells_bgk' +
                   ' comma-separated floats')
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'initial_temperature\''
            raise KeyError(err)

        try:
            initial_velocity = (np.fromstring(param_dict['initial_velocity'],
                                              sep=',') * units.cm / units.s)
            if len(initial_velocity) != self.n_species * self.n_cells_bgk:
                raise ValueError
            self.initial_velocity = initial_velocity.reshape((self.n_cells_bgk,
                                                              self.n_species))
            logging.debug('initial bulk velocity: ' +
                          np.array_str(self.initial_velocity))
        except ValueError:
            err = ('\'velocity\' must be n_species x n_cells_bgk' +
                   ' comma-separated floats')
            raise ValueError(err)
        except KeyError:
            err = 'missing parameter \'initial_velocity\''
            raise KeyError(err)


    def setup(self):
        '''clean up old bgk output and set up the needed folders for the md and 
        bgk simulations
        '''

        # set up folders for md
        logging.debug('creating folders for md')
        if not os.path.exists(self.testcase + '_md'):
            os.mkdir(self.testcase + '_md')
        os.chdir(self.testcase + '_md')

        # clean up files for bgk
        if self.only_md:
            return
        logging.debug('removing old BGK files with same name as test case')
        [os.remove(os.path.join(self.bgk_path, 'Data', f)) for f in
         os.listdir(self.bgk_path) if f.startswith(self.testcase + '_')]
        if os.path.exists(os.path.join(self.bgk_path, 'input', self.testcase)):
                os.remove(os.path.join(self.bgk_path, 'input', self.testcase))

    def initialize(self):
        ''' initialize all the distributions and macro properties that we
        need to track throughout the hmm
        '''

        # initialize the distributions
        logging.debug('initializing discrete distributions')
        self.distribution = np.empty((self.n_cells_bgk, self.n_species),
                                     dtype=object)
        for cell in range(self.n_cells_bgk):
            # compute mixture temperature for sufficiently wide velocity grid
            mixture_temperature = (np.sum(self.initial_temperature[cell,:] * 
                                          self.initial_density[cell,:]) /
                                   self.initial_density[cell,:].sum())
            for sp in range(self.n_species):
                # set grid based on maximum of mixture and species temperature
                T_use = max(mixture_temperature,
                            self.initial_temperature[cell,sp])
                v_thermal = np.sqrt(T_use / self.mass[sp])
                vx = vy = vz = np.linspace(-8. * v_thermal, 8. * v_thermal,
                                           self.n_vel)
                n = self.initial_density[cell,sp]
                m = self.mass[sp]
                u = np.array([self.initial_velocity[cell,sp], 0., 0.])
                ke = self.initial_temperature[cell,sp] * 3./2.
                f = distributions.discrete_maxwellian3D(vx, vy, vz, m, n, u, ke)
                self.distribution[cell,sp] = \
                       distributions.linear_interpolated_rv_3D(vx, vy, vz, f, m)

        # initialize taus and errors
        logging.debug('intializing other data variables')
        if self.only_md:
            n_bgks = 0
        else:
            n_bgks = int(np.rint(self.final_time / (self.timestep_bgk * 
                                                    self.tau_update_rate)))
        self.tau_hist = np.empty((n_bgks, self.n_cells_bgk, self.n_species,
                                  self.n_species))
        self.tau_error_hist = np.empty((n_bgks, self.n_cells_bgk,
                                        self.n_species, self.n_species))
        self.taus = np.empty((self.n_cells_bgk, self.n_species, self.n_species))

        # initialize macro properties
        self.density = np.empty(self.initial_density.shape)
        self.velocity = np.empty(self.initial_velocity.shape)
        self.kinetic_energy = np.empty(self.initial_temperature.shape)
        for cell in range(self.n_cells_bgk):
            for sp in range(self.n_species):
                self.density[cell,sp] = self.distribution[cell,sp].density
                self.velocity[cell,sp] = \
                        self.distribution[cell,sp].momentum[0] / self.mass[sp]
                self.kinetic_energy[cell,sp] = \
                        self.distribution[cell,sp].kinetic_energy
        
        # time tracking stuff
        self.current_time = 0.0
        self.bgk_counter = 0


    def taus_from_md(self):
        '''launch MD simulation(s) given the simualtion parameters and the
        current physical state, and compute the taus from this data
        
        Run n_simulations_md simualtions for each bgk cell, and compute taus
        from the generated data.
        '''

        # set up required folders
        logging.info('Setting up for MD run %d\n' % (self.bgk_counter))
        rootdir = os.getcwd()
        path = 'md_runs_' + str(self.bgk_counter)
        if not os.path.exists(path):
            os.mkdir(path)
        for cell in range(self.n_cells_bgk):
            if not os.path.exists(path + '/cell' + str(cell)):
                os.mkdir(path + '/cell' + str(cell))
                
        # loop over cells and run the md simulations
        for cell in range(self.n_cells_bgk):
            logging.info('Running MD simulation in cell %d' % (cell))

            # enter the simulation directory
            os.chdir(os.path.join(path, 'cell' + str(cell)))
            md_io.setup_md_workspace()

            # compute screening length
            electron_density = np.sum(self.charge * self.density[cell,:])
            mixture_temperature = (np.sum(self.density[cell,:] *
                                          2./3. * self.kinetic_energy[cell,:]) /
                                   np.sum(self.density[cell,:]))
            screen_length = np.sqrt(mixture_temperature /
                                    (4. * np.pi * electron_density))
            logging.debug('using mixture temperature %f and screen length %f' %
                          (mixture_temperature, screen_length))

            # 3D bulk velocity
            bulk_velocity = np.zeros((self.n_species,3))
            bulk_velocity[:,0] = self.velocity[cell,:]

            # cell size depends on if doing small box equilibration first
            if self.small_box_equilibration > 0:
                cell_size = self.cell_length_md / 2.
                cutoff = self.cutoff_md / 2.
            else:
                cell_size = params.cell_length_md
                cutoff = params.cutoff_md

            # set up md parameters with the current state
            logging.debug('setting md parameters')
            md_params = md_io.md_parameters(
                    n_sims=self.n_simulations_md, cell_size=cell_size,
                    n_timesteps=self.n_timesteps_md, timestep=self.timestep_md,
                    movie_rate=self.movie_rate, n_species=self.n_species,
                    density=self.density[cell,:], mass=self.mass,
                    charge=self.charge, screen_length=screen_length,
                    cutoff=cutoff, kinetic_energy=self.kinetic_energy[cell,:],
                    bulk_velocity=bulk_velocity, friction=self.friction[0],
                    n_mts_timesteps=self.mts_timesteps_md,
                    mts_cutoff=self.mts_cutoff_md,
                    mts_threshold=self.mts_threshold_md)
                    
            # equilibration phase
            if self.small_box_equilibration > 0:
                logging.debug('equilibrating on the small box')
                # first equilibrate on a smaller box, if specified
                md_params.equilibration_time = self.small_box_equilibration

                # set the parameters and phase space
                md_io.set_md_parameters(md_params, md)
                species, pos0, vel0 = md_io.generate_md_phasespace(md_params)
                md_io.set_md_phasespace(pos0, vel0, md)

                # equilibration
                md_io.equilibrate_md(md_params, md, print_rate=1000,
                                     save_rate=self.md_save_rate)
                pos1, vel1 = md_io.get_md_phasespace(md)

                # expand the box and set up parameters
                md_io.expand_md_box(md_params, md)

            # first equilibration on full box
            logging.debug('first full-size equilibration')
            md_params.equilibration_time = self.equilibration_time[0]
            md_io.equilibrate_md(md_params, md, print_rate=100,
                                 save_rate=self.md_save_rate)

            # second equilibration on full box
            logging.debug('second full-size equilibration')
            md_params.change_friction(self.friction[1], md)
            md_io.equilibrate_md(md_params, md, print_rate=100,
                                 save_rate=self.md_save_rate)
            md.closefiles()
    
            # run the simulations
            logging.debug('starting simulation phase')
            md.openfiles()
            energy, data = md_io.simulate_md(md_params,
                                             self.distribution[cell,:], md,
                                             print_rate=100,
                                             save_rate=self.md_save_rate)
            # check the dHdt
            timesteps_elapsed = self.n_timesteps_md
            while timesteps_elapsed < self.max_timesteps_md:
                mean = data.dHdt[0].mean(axis=0)
                std = data.dHdt[0].std(axis=0, ddof=1)
                (lb, ub) = stats.norm.interval(
                    0.95, loc=mean, scale=std/np.sqrt(timesteps_elapsed))
                logging.debug('mean: \n' + np.array_str(mean))
                logging.debug('std: \n' + np.array_str(std))
                logging.debug('lower bound: \n' + np.array_str(lb)) 
                logging.debug('upper bound: \n' + np.array_str(ub)) 
                if np.any(lb * ub < 0):
                    md_params.n_timesteps += self.md_save_rate
                    energy, data = md_io.simulate_md(
                        md_params, self.distribution[cell,:], md,
                        print_rate=100, resample=False,
                        save_rate=self.md_save_rate,
                        resume=True, last_step=timesteps_elapsed)
                    timesteps_elapsed += self.md_save_rate
                else:
                    break

            md.closefiles()

            # save data for auditing
            np.save('energy', energy)
            np.save('data.momentum', data.momentum)
            np.save('data.stress', data.stress)
            np.save('data.kinetic_energy', data.kinetic_energy)
            np.save('data.heat', data.heat)
            np.save('data.m5', data.m4)
            np.save('data.dHdt', data.dHdt)
            np.save('data.mass', data.mass)
            np.save('data.time', data.time)

            # compute the taus
            logging.info('Computing taus in cell %d\n' % (cell))
            dHdt = data.dHdt.mean(axis=(0,1))
            logging.debug('dHdt for the species is \n' + np.array_str(dHdt))
            (self.taus[cell,:,:],
             self.tau_error_hist[self.bgk_counter,cell,:,:]) = \
                    tau_utils.compute_taus(md_params, self.distribution[cell,:],
                                           dHdt)
            self.taus /= units.s
            self.tau_hist[self.bgk_counter,:,:,:] = self.taus
            logging.debug('taus are: \n' + np.array_str(self.taus))

            # leave the simulation directory
            os.chdir(rootdir)

    def bgk_step(self):
        '''run the bgk simulation with the md informed taus for the desired
        period of time and update the current state
        '''

        logging.info('Running BGK simulation %d at time %f' %
                     (self.bgk_counter, self.current_time))
        # set the parameters
        logging.debug('setting bgk parameters')
        bgk_params = bgk_io.bgk_parameters(
                case=self.testcase, n_dims=self.n_dim,
                length=self.cell_length_bgk, n_cells=self.n_cells_bgk,
                n_vel=self.n_vel, timestep=self.timestep_bgk,
                final_time=self.timestep_bgk*self.tau_update_rate,
                order=self.order, implicit=self.implicit,
                data_rate=self.data_rate_bgk, n_species=self.n_species,
                charge=self.charge, distribution=self.distribution,
                taus=self.taus, bgk_path=self.bgk_path)

        # run the simulation
        bgk_io.run_bgk_simulation(bgk_params)

        # load the distributions
        logging.info('Loading output data from the BGK simulation\n')
        if self.n_dim is 0:
            self.distribution[0,:] = bgk_io.read_distributions0D(bgk_params)

    
    def update_conditions(self):
        '''update the simulation conditions with the data in the distributions
        '''

        # update denstiy, temperature, velocity
        for cell in range(self.n_cells_bgk):
            for sp in range(self.n_species):
                self.density[cell,sp] = self.distribution[cell,sp].density
                self.velocity[cell,sp] = \
                        self.distribution[cell,sp].momentum[0] / self.mass[sp]
                self.kinetic_energy[cell,sp] = \
                        self.distribution[cell,sp].kinetic_energy

        # update time
        self.current_time += self.timestep_bgk * self.tau_update_rate
        self.bgk_counter += 1


    def march_simulation(self):
        '''run a "step" in the hmm procedure, defined by:

        1) run the MD simulation and compute taus
        2) run the bgk simulation using computed taus
        3) update the simulation conditions with the bgk output
        '''

        logging.info('marching simulation at time %.3e of %.3e\n' %
              (self.current_time, self.final_time))
        # run the MD simulations and get the taus
        self.taus_from_md()

        # run the bgk simulation
        self.bgk_step()

        # update conditions
        logging.info('Updating simulation parameters with BGK output')
        self.update_conditions()

    def run_hmm(self):
        ''' run an hmm simulation by looping over time to march the simulation,
        alternating betweeen md and bgk, until simulation end time is reached
        '''

        # loop over time
        while self.current_time < self.final_time:
            self.march_simulation()

    def run_md(self):
        ''' run just an MD simulation, basically a wrapper for
        md_io.simulate_md that also does equilibration
        '''

        # set up workspace
        rootdir = os.getcwd()
        path = 'md_runs_' + str(self.bgk_counter)
        if not os.path.exists(path):
            os.mkdir(path)
        for cell in range(self.n_cells_bgk):
            if not os.path.exists(path + '/cell' + str(cell)):
                os.mkdir(path + '/cell' + str(cell))
                
        # loop over cells and run the md simulations
        for cell in range(self.n_cells_bgk):
            logging.info('Running MD simulation in cell %d' % (cell))

            # enter the simulation directory
            os.chdir(os.path.join(path, 'cell' + str(cell)))
            md_io.setup_md_workspace()

            # compute screening length
            electron_density = np.sum(self.charge * self.density[cell,:])
            mixture_temperature = (np.sum(self.density[cell,:] *
                                          2./3. * self.kinetic_energy[cell,:]) /
                                   np.sum(self.density[cell,:]))
            screen_length = np.sqrt(mixture_temperature /
                                    (4. * np.pi * electron_density))
            logging.debug('using mixture temperature %f and screen length %f' %
                          (mixture_temperature, screen_length))

            # 3D bulk velocity
            bulk_velocity = np.zeros((self.n_species,3))
            bulk_velocity[:,0] = self.velocity[cell,:]

            # cell size depends on if doing small box equilibration first
            if self.small_box_equilibration > 0:
                cell_size = self.cell_length_md / 2.
                cutoff = self.cutoff_md / 2.
            else:
                cell_size = self.cell_length_md
                cutoff = self.cutoff_md

            # set up md parameters with the current state
            logging.debug('setting md parameters')
            md_params = md_io.md_parameters(
                    n_sims=self.n_simulations_md, cell_size=cell_size,
                    n_timesteps=self.n_timesteps_md, timestep=self.timestep_md,
                    movie_rate=self.movie_rate, n_species=self.n_species,
                    density=self.density[cell,:], mass=self.mass,
                    charge=self.charge, screen_length=screen_length,
                    cutoff=cutoff, kinetic_energy=self.kinetic_energy[cell,:],
                    bulk_velocity=bulk_velocity, friction=self.friction[0],
                    n_mts_timesteps=self.mts_timesteps_md,
                    mts_cutoff=self.mts_cutoff_md,
                    mts_threshold=self.mts_threshold_md,
                    n_procs=self.md_nprocs)
                    
            # equilibration phase
            if self.small_box_equilibration > 0:
                logging.debug('equilibrating on the small box')
                # first equilibrate on a smaller box, if specified
                md_params.equilibration_time = self.small_box_equilibration
    
                # set the parameters and phase space
                md_io.set_md_parameters(md_params, md)
                species, pos0, vel0 = md_io.generate_md_phasespace(md_params)
                md_io.set_md_phasespace(pos0, vel0, md)
    
                if not self.md_resume:
                    # equilibration
                    md_io.equilibrate_md(md_params, md, print_rate=1000,
                                         save_rate=self.md_save_rate)
                    pos1, vel1 = md_io.get_md_phasespace(md)
    
                # expand the box and set up parameters
                md_io.expand_md_box(md_params, md)
    
            # first equilibration on full box
            if self.equilibration_time[0] > 0 and not self.md_resume:
                logging.debug('first full-size equilibration')
                md_params.equilibration_time = self.equilibration_time[0]
                md_io.equilibrate_md(md_params, md, print_rate=100,
                                     save_rate=self.md_save_rate)
    
            # second equilibration on full box
            if self.equilibration_time[1] > 0 and not self.md_resume:
                logging.debug('second full-size equilibration')
                md_params.change_friction(self.friction[1], md)
                md_io.equilibrate_md(md_params, md, print_rate=100,
                                     save_rate=self.md_save_rate)
            md.closefiles()
            md.openfiles()
 
            # run the simulations
            logging.debug('starting simulation phase')
            if not self.md_resume:
                pos, vel = md_io.get_md_phasespace(md)
                np.save('start_pos', pos)
            else:
                logging.info("reinitializing previous simulation phase space")
                pos = np.load('end_pos.npy')
                vel = np.load('end_vel.npy')
                md_io.set_md_phasespace(pos, vel, md)

            if self.tau_update_rate > 0 and not self.only_md:
                energy, data, distribution_log = md_io.simulate_md(
                    md_params, self.distribution[cell,:], md, print_rate=100,
                    resample=self.md_resample,
                    refresh_rate=self.tau_update_rate,
                    save_rate=self.md_save_rate,
                    resume=self.md_resume,
                    last_step=self.md_last_step)
            else:
                energy, data = md_io.simulate_md(
                    md_params, self.distribution[cell,:], md, print_rate=100,
                    resample=self.md_resample,
                    refresh_rate=0,
                    save_rate=self.md_save_rate,
                    resume=self.md_resume,
                    last_step=self.md_last_step)


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
            pos, vel = md_io.get_md_phasespace(md)
            np.save('end_pos', pos)
            np.save('end_vel', vel)
            
            # leave the simulation directory
            os.chdir(rootdir)

