# plasma-bgkmd

## Dependencies

### Python:

1) numpy 

2) scipy

3) f2py 

### Other stuff:

1) gfortran compiler

2) Vlasov-BGK code (being open-sourced separately)
Note: the molecular dynamics and collision rate computation components will
work independently of the VBGK, but running a heterogeneous multiscale method
simulation requires the separate code.

## Setup and compiling

1) cd into the `bgk-md` repository
    
    cd bgk-md

2) add the `bgk-md` folder to your `$PYTHONPATH`
    
    export PYTHONPATH=`pwd`
    (or add to .bashrc file)

3) cd into the `hmm` folder
    
    cd hmm 

4) compile the MD fortran code using f2py

    f2py -m md --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c ../md/md.f90


## HMM input file
**Note: the list of inputs below is somewhat outdated. See `hmm.py` for a more complete list.**

### Syntax

The HMM code takes in an input file with all the parameters. The format is 

    paramter_name : parameter_value

Comment lines start with a # symbol, for example

    # this is a comment    
    
Parameters that take a list or array input should have
comma-separated values, in row-major format. Whitespace is ignored.
These can be broken up over multiple lines as long as each line ends with a 
comma. Comments in the middle of a parameter specification are not allowed.

For example, these are valid:

    parameter : 1, 2, 3, 4, 5, 6, 7, 8, 9
    parameter : 1, 2,
    3,4,5,6,7,8,9
    parameter : 1, 2, 3,
                4, 5, 6,
                7, 8, 9
                
But these are not:

    parameter : 1, 2,  3
    , 4, 5, 6, 7, 8, 9
    parameter : 1, 2, 3,
    # some comment
    4, 5, 6, 7, 8, 9
    
### Parameters

The following parameters are mostly required, with the exception of the cell a
few, which have defaults noted. Order does not matter. See the `hmm_basic_test`
for an example input file.

#### Testcase information
* `testcase` : name of test case
* `only_md` : flag to only do an MD simulation instead of a full HMM,
alleviates the need to set any of the BGK parameters except `n_vel`, although
one can set the distribution refresh rate via `tau_update_rate`

#### Species information
* `n_species` : number of species
* `mass` : mass of a particle in each species (in grams)
* `charge` : charge of a particle in each species (multiple of electron charge)
* `initial_density` : the initial density of each species in each cell (per cc)
* `initial_temperature` : the initial temperature of each species in each cell 
(in eV)
* `initial_velocity` : the initial drift velocity (x-direction)  of each
species in each cell (in cm/s)

Note: initial values should have a value for each species in each BGK cell. One
value per species if only\_md is set.

#### Macroscale (BGK) parameters
* `n_dim` : number of spatial dimensions (0 or 1)
(only 0D is  supported currently)
* `n_cells_bgk` : number of BGK cells in space
(will be ignored and automatically set to 1 if 0D)
* `n_vel` : number of velocity gridpoints in each direction (uniform grid)
* `cell_length_bgk` : length of each BGK cell (ignored in 0D)
* `order` : whether to use first or second order (1 or 2) (2 not tested), 
default is 1
* `implicit` : whether to use implicit time stepping (not tested)
* `timestep_bgk` : time step to use in the bgk simulations (in seconds)
* `final_time` : end simulation time (in seconds)
* `tau_update_rate` : how many timesteps to take between each update of the
relaxation rates
* `data_rate_bgk` : how many timesteps between writing the BGK simulation
outputs to file, default is 1
* `bgk_path` : path the the root directory of the BGK code

#### Microscale (MD) inputs
* `n_simulations_md` : number of MD simulation trials for ensemble averaging
when gathering data to compute relaxation times
* `timestep_md` : timestep to take in each MD simulation as proportion of
shortest plasma period (timestep is timestep\_md / f\_p)
* `friction` : how powerful the thermostat is, as fraction of plasma frequency
* `n_timesteps_md` : number of timesteps to take in the MD simulations
* `max_timesteps_md` : max number of timesteps for MD simulation if the
statistics are bad (default is only take `n_timesteps_md` steps). Note: need 
`save_rate` % `n_timesteps_md` = 0 and `save_rate % max_timesteps_md` = 0.
* `equilibration_time` : equlibration times in terms of slowest plasma period
for the strong and weak thermostat phases
* `small_box_equilibration` : equilibration time in terms of slowest plasma
period on a box half the size in each direction (1/8 the particles) for
speeding up equilibration. Step is skipped if set to zero.
* `cell_length_md` : length of an MD simulation cell as proportion of the
electron screening length, must be greater than double `cutoff_md`
* `cutoff_md` : cutoff radius for computing forces as proportion of the
electron screening length
* `mts_threshold_md` : threshold relative jump in total energy to trigger
redoing a time step using small time steps on the particles that are causing
conservation problems, default is to turn MTS off by setting to 100
* `mts_cutoff_md` : cutoff radius around the particles exerting most force on
each other over which to apply small time step, default is 0.0
* `mts_timesteps_md` : number of time steps to take over the course of one
regular time step for the labeled particles, default is 100
* `movie_rate` : how often to save data for an ovito movie (I think this is
somewhat broken right now and the movies will overwrite themselves if doing
multiple simulations)
* `md_save_rate` : how often to save MD phasespace and data for future resume
* `md_resume` : whether resuming from previous simulation, default is false
* `md_last_step` : last step taken to resume from, default is 0
* `md_resample` : whether or not to resample velocity distribution
after equilibrating the MD simulation
* `md_nprocs` : how mnay processors to use for the MD simulation (openMP
parallelized). The default is to use all available cores.
