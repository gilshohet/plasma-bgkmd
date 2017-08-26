'''Unit conversions to and from atomic units.

Converts units to other units by multiplying by the unit and from other units
by dividing.

Example: you have a velocity in SI units and want atomic units
    v_au = v_SI * meter / second

Example: you have a velocity in atomic units and want SI
    v_SI = v_au * second / meter
    
Example: you have a velocity in cgs and want SI
    v_SI = v_cgs * meter / centimeter
'''

from numpy import pi


# fundamental units and constants needed for conversions (in SI)
_electron_mass = 9.10938291e-31
_proton_mass = 1.672621898e-27
_elementary_charge = 1.602176565e-19
_planck_constant = 6.626070040e-34
_reduced_planck_constant = _planck_constant / (2.0 * pi)
_vacuum_permittivity = 8.854187817620e-12
_coulomb_force_constant = 1.0 / (4.0 * pi * _vacuum_permittivity)
_speed_of_light = 299792458
_boltzmann_constant = 1.38064852e-23
_avogadro = 6.022140857e23

# shorthand
_me = _electron_mass
_mp = _proton_mass
_e = _elementary_charge
_h = _reduced_planck_constant
_eps0 = _vacuum_permittivity
_ke = _coulomb_force_constant
_c = _speed_of_light
_kb = _boltzmann_constant


# NUMBER
mol = _avogadro

# MASS
kilogram = 1.0 / _me
gram = kilogram * 1.0e-3
milligram = kilogram * 1.0e-6
microgram = kilogram * 1.0e-9
nanogram = kilogram * 1.0e-12
picogram = kilogram * 1.0e-15
unified = gram / mol

# CHARGE
coulomb = 1.0 / _e

# LENGTH
meter = _me * _e**2 / (4.0 * pi * _eps0 * _h**2)
decimeter = 1.0e-1 * meter
centimeter = 1.0e-2 * meter
millimeter = 1.0e-3 * meter
micrometer = 1.0e-6 * meter
nanometer = 1.0e-9 * meter
angstrom = 1.0e-10 * meter
picometer = 1.0e-12 * meter

# VOLUME
liter = decimeter**3

# ENERGY
joule = (4.0 * pi * _eps0 * _h)**2 / (_me * _e**4)
erg = 1.0e-7 * joule
electronvolt = joule * _e

# TIME
second = 1 / (joule * _h)
millisecond = second * 1e-3
microsecond = second * 1e-6
nanosecond = second * 1e-9
picosecond = second * 1e-12
femtosecond = second * 1e-15

# FREQUENCY
hertz = 1.0 / second

# TEMPERATURE
kelvin = _kb * joule

# PRESSURE
pascal = joule / meter**3
barye = 1.0e-1 * pascal

# ELECTRIC POTENTIAL
volt = joule * _e

# FORCE
newton = joule / meter
dyne = 1.0e-5 * newton


# USEFUL CONSTANTS
au = 1.0
speed_of_light = _speed_of_light * meter / second
electron_mass = 1.0
proton_mass = _mp * kilogram
electron_charge = 1.0
atomic_mass_unit = unified
fine_structure_constant = _e**2 / (4 * pi * _eps0 * _h * _c)
bohr = 1.0
hartree = 1.0
avogadro = _avogadro
reduced_planck = 1.0
vacuum_permittivity = 1.0 / (4.0 * pi)

# SHORTHANDS
m_e = electron_mass
m_p = proton_mass
e = electron_charge
amu = atomic_mass_unit
alpha = fine_structure_constant
c = speed_of_light
a0 = bohr
E_h = hartree
h = reduced_planck
eps0 = vacuum_permittivity
kg = kilogram
g = gram
mg = milligram
ug = microgram
pg = picogram
C = coulomb
m = meter
dm = decimeter
cm = centimeter
mm = millimeter
nm = nanometer
A = angstrom
pm = picometer
L = liter
cc = cm**3
J = joule
eV = electronvolt
s = second
sec = second
ms = millisecond
us = microsecond
ns = nanosecond
ps = picosecond
fs = femtosecond
Hz = hertz
K = kelvin
Pa = pascal
Ba = barye
V = volt
N = newton
dyn = dyne
