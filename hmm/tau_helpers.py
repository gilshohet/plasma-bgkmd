''' helper functions that do all the messy gruntwork of computing taus

computes intraspecies tau from the dHdt, and also the interspecies taus using
a nonlinear least squares method to match the dHdt from MD as well as possible
'''

import distributions
import numpy as np
import scipy
import logging


def triple_integral(integrand, x, y, z):
    ''' compute the triple integral of a function using the trapezoidal rule
    given the x, y, z coordinates

    Parameters
    ----------
    integrand : 3D numpy array of function values
        the thing that you want to integrate
    x, y, z : 1D arrays of floats
        the coordinates corresponding to the values of the integrand

    Returns
    -------
    integral : float
        the result of the integration
    '''

    return np.trapz(np.trapz(np.trapz(integrand, z, axis=2), y, axis=1), x)


def compute_intraspecies_tau(distribution, dHdt):
    ''' compute the single species tau_kk

    Parameters
    ----------
    distribution : 3D distribution object
        the distribution function for this species
    dHdt : float
        average dHdt for this species acting on itself

    Returns
    -------
    tau : float
        single species relaxation parameter tau_kk
    f_eq : distribution
        equilibrium distribution relaxing towards
    '''

    # extract information
    m = distribution.mass
    n = distribution.density
    u = distribution.momentum / m
    KE = distribution.kinetic_energy
    vx = distribution._x
    vy = distribution._y
    vz = distribution._z
    f = distribution.distribution
    f_eq = distributions.discrete_maxwellian3D(vx, vy, vz, m, n, u, KE)

    # integrate and compute tau
    tau = triple_integral((f_eq - f) * np.log(f), vx, vy, vz) / dHdt

    return tau, f_eq

def compute_interspecies_tau(distribution1, distribution2, dHdt12, dHdt21, which_tau=2):
    ''' compute the interspecies relaxation parameters by a nonlinear least
    squares solve to find the optimal ratio of taus
    

    Parameters
    ----------
    distribution1, distribution2 : distribution objects
        the distributions of species 1 and 2
    dHdt12, dHdt21 : floats
        dHdt of species 1 due to species 2 and vice-versa, respectively
    which_tau : int
        which tau computation method to use

    Returns
    -------
    tau12, tau21 : floats
        relaxation time of species 1 due to species 2 and vice-versa
    error : array to two floats
        error from the taus
    f12, f21 : distributions
        interspecies equilibrium distributions being relaxed towards

    @TODO Fix analytical jacobian at some point
    '''
    
    # setup (extract data from distributions)
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
    
    # initial guess for tau ratio, assume temperature relaxation
#   tau_ratio0 = n2 / n1
#   tau_ratio0 = n2 * m2 / (n1 * m1)
#   tau_ratio0 = 0.5 * n2 / n1 * (1 + m2 / m1)
#   tau_ratio = tau_ratio0
    if which_tau is 0:
        tau_ratio0 = n2 * m2 / (n1 * m1)
    elif which_tau is 1:
        tau_ratio0 = n2 / n1
    else:
        tau_ratio0 = 0.5 * n2 / n1 * (1 + m2 / m1)

    # mixture quantitities
    f12, f21, u12, T12 = distributions.equilibrium_maxwellian3D(
            n1, n2, u1, u2, T1, T2, m1, m2, vx1, vy1, vz1, vx2, vy2, vz2,
            tau_ratio0, return_all=True)

    # dH in the kinetic sense
    dH12_bgk = (triple_integral(f12 * np.log(f1), vx1, vy1, vz1) - 
                triple_integral(f1 * np.log(f1), vx1, vy1, vz1))
    dH21_bgk = (triple_integral(f21 * np.log(f2), vx2, vy2, vz2) - 
                triple_integral(f2 * np.log(f2), vx2, vy2, vz2))

    # least squares functions
    def residual(tau_ratio):
        tau_ratio = tau_ratio[0]
        # mixture quantitities
        f12, f21, u12, T12 = distributions.equilibrium_maxwellian3D(
                n1, n2, u1, u2, T1, T2, m1, m2, vx1, vy1, vz1, vx2, vy2, vz2,
                tau_ratio, return_all=True)

        # dH in the kinetic sense
        dH12_bgk = (triple_integral(f12 * np.log(f1), vx1, vy1, vz1) - 
                    triple_integral(f1 * np.log(f1), vx1, vy1, vz1))
        dH21_bgk = (triple_integral(f21 * np.log(f2), vx2, vy2, vz2) - 
                    triple_integral(f2 * np.log(f2), vx2, vy2, vz2))
        tau_kl = ((dHdt12 * dH12_bgk + tau_ratio * dHdt21 * dH21_bgk) / 
                  (dHdt12**2 + dHdt21**2 * tau_ratio**2))
        resid =  np.array([(dHdt12 - dH12_bgk / tau_kl) / dHdt12,
                           (dHdt21 - dH21_bgk / (tau_kl * tau_ratio)) / dHdt21])
        logging.debug('residual: ' + np.array_str(resid))
        logging.debug('ratio of taus: %f' % tau_ratio)
        return resid

#   def jacobian(tau_ratio):
#       ''' ANALYTICAL JACOBIAN DOES NOT CURRENTLY SEEM TO WORK '''
#       tau_ratio = tau_ratio[0]

#       # mixture quantitities
#       f12, f21, u12, T12 = distributions.equilibrium_maxwellian3D(
#               n1, n2, u1, u2, T1, T2, m1, m2, vx1, vy1, vz1, vx2, vy2, vz2,
#               tau_ratio, return_all=True)
#       A12 = n1 * (m1 / (2. * np.pi * T12))**(3./2.)
#       A21 = n2 * (m2 / (2. * np.pi * T12))**(3./2.)
#       # dH in the kinetic sense
#       dH12_bgk = (triple_integral(f12 * np.log(f1), vx1, vy1, vz1) - 
#                   triple_integral(f1 * np.log(f1), vx1, vy1, vz1))
#       dH21_bgk = (triple_integral(f21 * np.log(f2), vx2, vy2, vz2) - 
#                   triple_integral(f2 * np.log(f2), vx2, vy2, vz2))
#       tau12 = (((dH12_bgk * dHdt21 * tau_ratio)**2 + (dH21_bgk * dHdt12)**2) /
#                (dHdt12 * dHdt21 * tau_ratio * 
#                 (dH12_bgk * dHdt21 * tau_ratio + dH21_bgk * dHdt12)))

#       # integrals
#       vsq1 = (vx1 - u12[0])**2 + (vy1 - u12[1])**2 + (vz1 - u12[2])**2
#       f1logf1 = triple_integral(f1 * np.log(f1), vx1, vy1, vz1)
#       f12logf1 = triple_integral(f12 * np.log(f1), vx1, vy1, vz1)
#       vf12logf1 = np.array([
#           triple_integral((vx1 - u12[0]) * f12 * np.log(f1), vx1, vy1, vz1),
#           triple_integral((vy1 - u12[1]) * f12 * np.log(f1), vx1, vy1, vz1),
#           triple_integral((vz1 - u12[2]) * f12 * np.log(f1), vx1, vy1, vz1)])
#       vsqf12logf1 = triple_integral(vsq1 * f12 * np.log(f1), vx1, vy1, vz1)
#       vsq2 = (vx2 - u12[0])**2 + (vy2 - u12[1])**2 + (vz2 - u12[2])**2
#       f2logf2 = triple_integral(f2 * np.log(f2), vx2, vy2, vz2)
#       f21logf2 = triple_integral(f21 * np.log(f2), vx2, vy2, vz2)
#       vf21logf2 = np.array([
#           triple_integral((vx2 - u12[0]) * f21 * np.log(f2), vx2, vy2, vz2),
#           triple_integral((vy2 - u12[1]) * f21 * np.log(f2), vx2, vy2, vz2),
#           triple_integral((vz2 - u12[2]) * f21 * np.log(f2), vx2, vy2, vz2)])
#       vsqf21logf2 = triple_integral(vsq2 * f21 * np.log(f2), vx2, vy2, vz2)
#           

#       # derivatives
#       du12 = ((n1 * n2 * m1 * m2) / (n1 * m1 * tau_ratio + n2 * m2)**2 *
#               (u1 - u2))
#       dT12 = (1. / (3. * tau_ratio * n1 + 3. * n2)**2 * 
#               (9. * n1 * n2 * (T1 - T2) +
#                3. * n1 * n2 * (m1 * np.sum(u1**2 - u12**2) -
#                                m2 * np.sum(u2**2 - u12**2))) - 
#               (2. * m1 * m2 * n1 * n2 * (tau_ratio * m1 * n1 * 
#                np.sum(u1**2 - u1 * u2) + m2 * n2 * np.sum(u1 * u2 - u2**2))) /
#               ((3. * tau_ratio * n1 + 3. * n2) * (tau_ratio * m1 * n1 +
#                m2 * n2)**2))
#       dA12 = (-3. * n1 / 2. * (m1 / (2. * np.pi))**(3./2.) *
#               T12**(-5./2.) * dT12)
#       dA21 = (-3. * n2 / 2. * (m2 / (2. * np.pi))**(3./2.) *
#               T12**(-5./2.) * dT12)
#       dtau12 = (dH21_bgk * ((dH12_bgk * dHdt21 * tau_ratio)**2 - 
#                  2 * tau_ratio * dH12_bgk * dH21_bgk * dHdt12 * dHdt21 - 
#                  (dH21_bgk * dHdt12)**2) /
#                 (dHdt21 * tau_ratio**2 * ((dH12_bgk * dHdt21 * tau_ratio)**2 + 
#                  2 * tau_ratio * dH12_bgk * dH21_bgk * dHdt12 * dHdt21 +
#                  (dH21_bgk * dHdt12)**2)))
#       dH12 = (f12logf1 * dA12 / A12 +
#                   np.sum(vf12logf1 * m1 / T12 * du12) +
#                   vsqf12logf1 * m1 / (2. * T12**2) * dT12)
#       dH21 = (f21logf2 * dA21 / A21 +
#                   np.sum(vf21logf2 * m2 / T12 * du12) +
#                   vsqf21logf2 * m2 / (2. * T12**2) * dT12)
#       dH12dt_bgk = f12logf1 - f1logf1
#       dH21dt_bgk = f21logf2 - f2logf2
#       dG1 = (1. / tau12**2 * dH12dt_bgk / dHdt12 * dtau12 -
#              1. / (tau12 * dHdt12) * dH12)
#       dG2 = ((1. / (tau_ratio**2 * tau12) +
#               1. / (tau_ratio * tau12**2) * dtau12) * dH21dt_bgk / dHdt21 - 
#              1. / (tau_ratio * tau12 * dHdt21) * dH21)
#       dG = np.array([dG1, dG2]).reshape((2,1))
#       return dG
    
    if which_tau is 2:
        logging.debug('doing least squares optimization')
        temperature_tau_ratio = n2 / n1
        momentum_tau_ratio = (m2 * n2) / (m1 * n1)
        lb = min(temperature_tau_ratio, momentum_tau_ratio)
        ub = max(temperature_tau_ratio, momentum_tau_ratio)
        output = scipy.optimize.least_squares(residual, tau_ratio0, #jac=jacobian,
                                              bounds=(lb, ub),
                                              ftol=1e-6, xtol=1e-6, gtol=1e-6)
        tau_ratio = output.x[0]
        error = abs(output.fun)
        logging.debug('least squares terminated due to: ' + output.message)
    else:
        tau_ratio = tau_ratio0
        error = np.array([-1, -1])

    # mixture quantitities
    f12, f21, u12, T12 = distributions.equilibrium_maxwellian3D(
            n1, n2, u1, u2, T1, T2, m1, m2, vx1, vy1, vz1, vx2, vy2, vz2,
            tau_ratio, return_all=True)
    
    # dH in the kinetic sense
    dH12_bgk = (triple_integral(f12 * np.log(f1), vx1, vy1, vz1) - 
                triple_integral(f1 * np.log(f1), vx1, vy1, vz1))
    dH21_bgk = (triple_integral(f21 * np.log(f2), vx2, vy2, vz2) - 
                triple_integral(f2 * np.log(f2), vx2, vy2, vz2))
    tau12 = ((dHdt12 * dH12_bgk + tau_ratio * dHdt21 * dH21_bgk) / 
              (dHdt12**2 + tau_ratio**2 * dHdt21**2))
    tau21 = tau_ratio * tau12


    return tau12, tau21, error, f12, f21

