'''
compute the moments, either for a discrete distribution or set of particles

'''

from scipy.integrate import trapz
import numpy as np

def distribution_moments(vx, vy, vz, distribution, mass=1):
    ''' compute the moments of a discrete velocity distribution, normalized by
    density so quantities are per particle

    Parameters
    ----------
    vx, vy, vz : discrete x, y, z velocities, meshgrid formatted or 1D arrays
    distribution: 3D array of the distribution function at each velocity
    mass : mass of a particle

    Returns
    -------
    density : float, the number density corresponding to the distribution
    momentum : 1x3 float, momentum per particle
    stress : 3x3 float, stress tensor per particle
    kinetic_energy : float, kinetic energy per particle
    heat : heat tranfer per particle
    m4 : fourth moment of the distribution (v**4)
    '''

    if vx.ndim is not 3:
        vx, vy, vz = np.meshgrid(vx, vy, vz, indexing='ij')

    density = trapz(trapz(trapz(distribution, vz, axis=2), vy[:,:,0], axis=1),
                    vx[:,0,0], axis=0)
    if density == 0.0:
        return 0.0, np.zeros((3,)), np.zeros((3,3)), 0.0, np.zeros((3,)), 0.0
    x_momentum = trapz(trapz(trapz(vx*distribution, vz, axis=2),
                       vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    y_momentum = trapz(trapz(trapz(vy*distribution, vz, axis=2),
                       vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    z_momentum = trapz(trapz(trapz(vz*distribution, vz, axis=2), 
                       vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    wx = vx - x_momentum / mass
    wy = vy - y_momentum / mass
    wz = vz - z_momentum / mass
    v2 = wx**2 + wy**2 + wz**2
    stress_xx = trapz(trapz(trapz(wx*wx*distribution, vz, axis=2),
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    stress_yy = trapz(trapz(trapz(wy*wy*distribution, vz, axis=2), 
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    stress_zz = trapz(trapz(trapz(wz*wz*distribution, vz, axis=2),
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    stress_xy = trapz(trapz(trapz(wx*wy*distribution, vz, axis=2), 
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    stress_xz = trapz(trapz(trapz(wx*wz*distribution, vz, axis=2),
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    stress_yz = trapz(trapz(trapz(wy*wz*distribution, vz, axis=2),
                      vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    kinetic_energy = trapz(trapz(trapz(v2*distribution, vz, axis=2),
                   vy[:,:,0], axis=1), vx[:,0,0], axis=0) * 0.5 * mass / density
    x_heat = trapz(trapz(trapz(wx*v2*distribution, vz, axis=2),
                   vy[:,:,0], axis=1), vx[:,0,0], axis=0) * 0.5 * mass / density
    y_heat = trapz(trapz(trapz(wy*v2*distribution, vz, axis=2),
                   vy[:,:,0], axis=1), vx[:,0,0], axis=0) * 0.5 * mass / density
    z_heat = trapz(trapz(trapz(wz*v2*distribution, vz, axis=2),
                   vy[:,:,0], axis=1), vx[:,0,0], axis=0) * 0.5 * mass / density
    m4 = trapz(trapz(trapz(v2*v2*distribution, vz, axis=2),
               vy[:,:,0], axis=1), vx[:,0,0], axis=0) * mass / density
    
    momentum = np.array([x_momentum, y_momentum, z_momentum])
    stress = np.array([[stress_xx, stress_xy, stress_xz],
                       [stress_xy, stress_yy, stress_yz],
                       [stress_xz, stress_yz, stress_zz]])
    heat = np.array([x_heat, y_heat, z_heat])
    
    return density, momentum, stress, kinetic_energy, heat, m4


def particle_moments(velocities, mass=1.0):
    ''' compute the moments (per particle) based on the velocites

    Parameters
    ----------
    velocities : n_particles x 3 array of floats, velocity of each particle
    mass : mass of a particle

    Returns
    -------
    momentum : 1x3 float, momentum per particle
    stress : 3x3 float, stress tensor per particle
    kinetic_energy : float, kinetic energy per particle
    heat : heat tranfer density per particle
    m4 : fourth moment of the distribution (v**4)
    '''

    N = velocities.shape[0]
    vx = velocities[:,0]
    vy = velocities[:,1]
    vz = velocities[:,2]
    momentum = np.sum(mass * velocities, axis=0) / N
    wx = vx - momentum[0] / mass
    wy = vy - momentum[1] / mass
    wz = vz - momentum[2] / mass
    v2 = wx**2 + wy**2 + wz**2
    stress = np.array([[np.sum(wx*wx), np.sum(wx*wy), np.sum(wx*wz)],
                       [np.sum(wx*wy), np.sum(wy*wy), np.sum(wy*wz)],
                       [np.sum(wx*wz), np.sum(wy*wz), np.sum(wz*wz)]]) 
    stress *= float(mass) / (N-1)
    kinetic_energy = np.sum(0.5 * mass * v2) / (N-1)
    heat = 0.5 * mass * np.array([np.sum(wx*v2), np.sum(wy*v2), 
                                  np.sum(wz*v2)]) / (N-1)
    m4 = np.sum(v2*v2) * mass / (N-1)
    
    return momentum, stress, kinetic_energy, heat, m4
