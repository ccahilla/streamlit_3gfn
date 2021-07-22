"""
future_detector_freq_noise_budget.py

Makes the future input mode cleaner frequency noise budget.
Reproduces Figure 5 for the paper, from ./Mathematica/IMCnoise.nb
"""

# import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as scc

import streamlit as st

mpl.rcParams.update({   'figure.figsize': (12, 9),
                        'text.usetex': True,
                        'font.family': 'serif',
                        # 'font.serif': 'Georgia',
                        # 'mathtext.fontset': 'cm',
                        'lines.linewidth': 2.5,
                        'font.size': 20,
                        'xtick.labelsize': 'large',
                        'ytick.labelsize': 'large',
                        'legend.fancybox': True,
                        'legend.fontsize': 16,
                        'legend.framealpha': 0.9,
                        'legend.handletextpad': 0.5,
                        'legend.labelspacing': 0.2,
                        'legend.loc': 'best',
                        'legend.columnspacing': 2,
                        'savefig.dpi': 80,
                        'pdf.compression': 9})

PROJECT_URL = "https://github.com/ccahilla/streamlit_3gfn"
PAPER_URL = "https://git.ligo.org/3gfrequencynoise/3gfn-paper"

#####   Set up figures directory   #####
# script_path = os.path.dirname(os.path.abspath(__file__))
# script_name = os.path.basename(os.path.abspath(__file__)).replace('.py', '')
# git_dir = os.path.abspath(os.path.join(script_path, '..'))
# fig_dir = f'{git_dir}/figures/{script_name}'
# # print()
# # print('Figures made by this script will be placed in:')
# # print(fig_dir)
# if not os.path.exists(fig_dir):
#     os.makedirs(fig_dir)



#####   Functions   #####
def fsr(length):
    """Free spectral range, returned in Hz. Assumes two mirror cavity length in meters"""
    return scc.c/(2 * length)

def finesse(T1, T2, Art):
    """Cavity finesse, given the two mirror transmissions and round trip losses.
    Inputs:
    -------
    T1: float
        transmission of the input mirror
    T2: float
        transmission of the end mirror
    Art: float
        cavity round-trip losses
    """
    r1 = np.sqrt(1 - T1)
    r2 = np.sqrt(1 - T2 - Art)
    FF = np.pi * np.sqrt(r1 * r2) / (1 - r1 * r2)
    return FF

def cavity_pole(length, T1, T2, Art):
    """Cavity finesse, given the two mirror transmissions and round trip losses.
    Inputs:
    -------
    length: float
        length of the cavity
    T1: float
        transmission of the input mirror
    T2: float
        transmission of the end mirror
    Art: float
        cavity round-trip losses

    Output:
    -------
    pole: float
        cavity pole in Hz
    """
    r1 = np.sqrt(1 - T1)
    r2 = np.sqrt(1 - T2 - Art)
    fsr0 = fsr(length)
    pole = (fsr0 / (2 * np.pi)) * np.log(1/(r1 * r2))
    return pole

def circulating_power(Pin, finesse):
    """Circulating power in the cavity, in the high finesse limit 
    where cavity gain = finesse/pi.
    Inputs:
    -------
    Pin: float
        cavity input power in watts
    finesse: float
        cavity finesse

    Output:
    -------
    Pcav: float
        cavity circulating power in watts
    """
    return Pin * finesse / np.pi 

def beam_waist(length, lam=1064e-9):
    """Beam waist in meters given the cavity length in meters"""
    return np.sqrt(length * lam / np.pi)

def max_power_density(
    Pin,
    length,
    finesse,
    lam=1064e-9,    
):
    """Max power density inside a cavity, in W/cm^2"""
    Pcav = Pin * finesse / np.pi 
    power_density = 1e-4 * 2 * Pcav / (length * lam)
    return power_density 

### Noises
# radiation pressure noise
def radiation_pressure_length_noise(
    ff,
    Pin,
    finesse,
    mass=2.92,
    lam=1064e-9,
):
    """Radiation pressure length noise for a single optic in a cavity in units of m/rtHz.
    Inputs:
    -------
    ff: float
        audio frequency in Hz
    Pin: float
        cavity input power in watts
    finesse: float
        cavity finesse
    mass: float
        mass of the optics in kilograms. Default is 2.92 kg.
    lam: float
        wavelength of the laser in meters. Default is 1064 nm.

    Outputs:
    --------
    length_noise_rad_pressure: float
        radiation pressure noise in m/rtHz
    """
    Pcav = circulating_power(Pin, finesse) 
    length_noise_rad_pressure = 2 * np.sqrt(2 * scc.c * scc.h * Pcav / lam) / (scc.c * mass * ff**2)
    return length_noise_rad_pressure

def radiation_pressure_freq_noise(
    ff,
    Pin,
    length,
    finesse,
    mass=2.92,
    lam=1064e-9,
):
    """Radiation pressure frequency noise for a total cavity in units of Hz/rtHz.
    Inputs:
    -------
    ff: float
        audio frequency in Hz
    Pin: float
        cavity input power in watts
    length: float
        cavity single-trip length
    finesse: float
        cavity finesse
    mass: float
        mass of the optics in kilograms. Default is 2.92 kg.
    lam: float
        wavelength of the laser in meters. Default is 1064 nm.

    Outputs:
    --------
    freq_noise_rad_pressure: float
        radiation pressure noise in Hz/rtHz
    """
    fsr0 = fsr(length)
    length_noise_rad_pressure = radiation_pressure_length_noise(ff, Pin, finesse, mass=mass, lam=lam)
    freq_noise_rad_pressure = 4 * fsr0 * length_noise_rad_pressure / lam
    return freq_noise_rad_pressure

# Coating brownian noise
def beam_waist(length, lam=1064e-9):
    """Minimum beam waist given the cavity length"""
    return np.sqrt(length * lam / np.pi)

def coating_brownian_length_noise(
    ff,
    beam_radius,
    temperature=295,
    poisson_ratio=0.17,
    young_modulus=72e9,
    coating_thickness=4.5e-6,
    loss_angle=4e-4,
):
    """Coating brownian length noise for a single optic in a cavity in units of m/rtHz.
    Inputs:
    -------
    ff: float
        audio frequency in Hz
    beam_radius: float
        beam spot size on the optics
    temperature: float
        optic temperature in Kelvin
    poisson_ratio: float
        optic coating poisson ratio sigma. Default is 0.17.
    young_modulus: float
        optic Young's modulus. Default is 72e9.
    loss_angle: float
        optical coating loss angle. Default is 4e-4.

    Outputs:
    --------
    length_noise_coating_brownian: float
        coating brownian noise for a single optic in m/rtHz
    """
    kB = scc.Boltzmann
    foo = 4 * kB * temperature / (np.pi**2 * ff)
    bar = (1 + poisson_ratio) * (1 - 2 * poisson_ratio) / young_modulus
    baz = coating_thickness * loss_angle / beam_radius**2
    length_noise_coating_brownian = np.sqrt( foo * bar * baz )

    return length_noise_coating_brownian

def coating_brownian_freq_noise(
    ff,
    length,
    beam_radius,
    temperature=295,
    poisson_ratio=0.17,
    young_modulus=72e9,
    coating_thickness=4.5e-6,
    loss_angle=4e-4,
    lam=1064e-9,
):
    """Coating brownian frequency noise for an entire cavity in Hz/rtHz

    Inputs:
    -------
    ff: float
        audio frequency in Hz
    length: float
        cavity single-trip length in meters
    beam_radius: float
        beam spot size on the optics
    temperature: float
        optic temperature in Kelvin
    poisson_ratio: float
        optic coating poisson ratio sigma. Default is 0.17.
    young_modulus: float
        optic Young's modulus. Default is 72e9.
    loss_angle: float
        optical coating loss angle. Default is 4e-4.
    lam: float
        wavelength of the laser in meters. Default is 1064 nm.

    Outputs:
    --------
    freq_noise_rad_pressure: float
        coating brownian noise for a cavity in Hz/rtHz
    """
    length_noise_coating_brownian = coating_brownian_length_noise(
        ff, 
        beam_radius, 
        temperature=temperature, 
        poisson_ratio=poisson_ratio, 
        young_modulus=young_modulus, 
        coating_thickness=coating_thickness, 
        loss_angle=loss_angle
    ) # m/rtHz

    fsr0 = fsr(length)
    freq_noise_coating_brownian = 2 * fsr0 * np.sqrt(2 * length_noise_coating_brownian**2) / lam # Hz/rtHz
    return freq_noise_coating_brownian

def ce_coating_brownian_length_noise(ff, L_ce=40e3):
    """Cosmic Explorer 1um coating brownian length noise in m/rtHz
    """
    strain_asd = 2e-25/(ff/10)**0.5
    return strain_asd * L_ce

def ce_coating_brownian_freq_noise(ff, L_ce=40e3, lam=1064e-9):
    """Cosmic Explorer 1um coating brownian frequency noise in Hz/rtHz
    """
    nu0 = scc.c / lam
    ce_coatings_length_noise = ce_coating_brownian_length_noise(ff)
    ce_coatings_freq_noise = nu0 / L_ce * ce_coatings_length_noise
    return ce_coatings_freq_noise

# NPRO noise
def npro_freq_noise(ff):
    """Freerunning NPRO noise, 1/f with 100 Hz/rtHz at f = 100 Hz
    """
    return 1e4/ff

# VCO noise
def vco_freq_noise(ff):
    """Voltage controlled oscillators frequency noise, 
    measured to be about flat at 1e-2 Hz/rtHz
    """
    vco_flat_noise = 1e-2 # Hz/rtHz
    return vco_flat_noise * np.ones_like(ff)

# Sensing noise
def mod_depth_to_sideband_power(gamma):
    """Takes in modulation depth, returns the total sideband power: p = 2 (gamma^2/4)
    """
    sideband_power = gamma**2 / 2
    return sideband_power

def sideband_power_to_mod_depth(sideband_power):
    """Takes in sideband power, returns the modulation depth: p = 2 (gamma^2/4)
    """
    gamma = np.sqrt(2 * sideband_power)
    return gamma

def shot_noise(
    power, 
    gamma,
    epsilon,
    lam=1064e-9
):
    """Calculates detector shot noise in W/rtHz, including non-signal junk light ratio eplison
    and cyclostationary increased shot noise due to sidebands with modulation depth gamma
    """
    c = scc.c
    h = scc.Planck
    shot = np.sqrt((4 * power * c * h / lam) * ((3/2) * (gamma**2/2) + epsilon ))
    return shot

def pdh_signal(
    ff,
    power,
    gamma,
    finesse,
    FSR,
    fpole
):
    """Laser frequency Pound-Drever-Hall signal response to a cavity in W/Hz.
    """
    pdh = 2 * power * gamma * finesse / FSR / (1 + 1j * ff / fpole)
    return pdh

def sensing_noise(
    ff,
    power,
    finesse,
    FSR,
    fpole,
    lam=1064e-9
):
    """Pound-Drever-Hall frequency sensing noise, Hz/rtHz,
    assuming epsilon = 2/gamma**2
    """
    c = scc.c
    h = scc.Planck
    sense = (FSR / finesse) * np.sqrt((5 * c * h) / (4 * power * lam)) * zero(ff, fpole)
    return sense

# CE freq noise requirement
def zero(ff, zero_hz):
    """Returns a transfer function of a zero at zero_hz 
    on a frequency vector ff
    """
    zero_tf = 1 + 1j * ff/zero_hz
    return zero_tf

def pole(ff, pole_hz):
    """Returns a transfer function of a pole at pole_hz 
    on a frequency vector ff
    """
    pole_tf = 1/(1 + 1j * ff/pole_hz)
    return pole_tf

def requirement_ce_freq_noise(ff, ce_freq_noise_mag_req=7e-7, ce_darm_pole=825):
    """Requirement for the CE frequency noise, 
    informed by the Advanced LIGO coupling frequency noise limits.
    Inputs:
    -------
    ff: float(s)
        frequency vector 
    ce_freq_noise_mag_req: float
        frequency noise requirement at DC in Hz/rtHz
    ce_darm_pole: float
        DARM pole of CE in Hz
    """
    requirement = ce_freq_noise_mag_req * np.abs(zero(ff, ce_darm_pole))
    return requirement

# Controls suppression
def fss(ff, stages, bandwidth=500e-3):
    """Frequency stabilization servo, which defines the frequency noise suppression
    Inputs:
    -------
    ff: array of floats
        frequency vector
    stages: int
        number of stages of boosts in the suppression
    bandwidth: float
        control loop bandwidth frequency in Hz
    
    Output: 
    -------
    suppression: array of floats
        frequency noise suppression due to the fss at each stage
    """
    suppression = (bandwidth/(1j * ff))**(stages + 1) \
                * (zero(ff, bandwidth/5) / np.abs(1 + 5j))**stages \
                * np.abs(1 + 1j/5) \
                * pole(ff, 5 * bandwidth)
    return suppression

def get_cavity_reflectivity_from_finesse(finesse):
    """Takes in cavity finesse, returns mirror power reflectivity R = r1 * r2
    """
    RR = ( (np.sqrt(4 * finesse**2 + np.pi**2) - np.pi) / (2 * finesse))**2
    return RR

def cavity_low_pass(ff, RR, FSR):
    """Defines the cavity low pass filter: C_LP(f) = (1-R)/(1 - R e^{i 2 pi f / FSR})
    """
    return (1 - RR) / (1 - RR * np.exp(-1j * 2 * np.pi * ff / FSR ) )

def cavity_high_pass(ff, RR, FSR):
    """Defines the cavity high pass filter: C_HP(f) = 1 - (1-R)/(1 + i ff / fpole)
    """
    clp = cavity_low_pass(ff, RR, FSR)
    return 1 - clp

def cavity_low_pass_approx(ff, fpole):
    """Defines the cavity low pass filter approx: C_LP(f) ~ 1/(1 + i ff / fpole)
    """
    return 1/(1 + 1j * ff / fpole)

def cavity_high_pass_approx(ff, fpole):
    """Defines the cavity high pass filter approx: C_HP(f) ~ i ff / fpole/(1 + i ff / fpole)
    """
    return (1j * ff / fpole)/(1 + 1j * ff / fpole)

def coupled_cavity_low_pass(ff, r1, r2, r3, L1, L2):
    """Defines the coupled cavity reflection low pass filter: CC_LP(f)
    """
    c = scc.c
    t1 = np.sqrt(1 - r1**2)
    t2 = np.sqrt(1 - r2**2)
    numer = np.exp((2j * ff * np.pi * (L1 + L2))/c) * r3 * t1 * t2
    denom = 1   + r1 * r2 * np.exp((4j * ff * L1 * np.pi)/c) \
                - r2 * r3 * np.exp((4j * ff * L2 * np.pi)/c) \
                - r1 * r3 * (r2**2 + t2**2) * np.exp((4j * ff * (L1 + L2) * np.pi)/c) 
    return numer / denom

def coupled_cavity_high_pass(ff, r1, r2, r3, L1, L2):
    """Defines the cavity high pass filter: C_HP(f) = 1 - (1-R)/(1 + i ff / fpole)
    """
    clp = coupled_cavity_low_pass(ff, r1, r2, r3, L1, L2)
    return 1 - clp

def coupled_cavity_pole(r1, r2, r3, L2):
    """Coupled cavity pole in Hz
    """
    c = scc.c
    t2 = np.sqrt(1 - r2**2)
    fcc = c / (4 * np.pi * L2) * np.log((1 + r1 * r2)/(r2 * r3 + r1 * r3 * (r2**2 + t2**2)))
    return fcc

def loop_noise_coupling_functions(olg, clp, chp, lock='active'):
    """Control loop coupling functions for a opto-mechanical cavity.
    Inputs:
    -------
    olg: array of complex floats
        Open loop gain
    clp: array of complex floats
        Cavity low pass
    chp: array of complex floats
        Cavity high pass
    lock: string
        Either "active", representing locking the laser to the cavity,
        or "passive", representing locking the cavity to the laser.

    Outputs:
    --------
    a_in: array of complex floats
        Input frequency noise coupling to transmitted frequency noise
    a_sense: array of complex floats
        Sensing frequency noise coupling to transmitted frequency noise
    a_disp: array of complex floats
        Displacement frequency noise coupling to transmitted frequency noise
    """
    if lock == 'active':
        a_in = clp / (1 + clp * olg)
        a_sense = clp * olg / (1 + clp * olg)
        a_disp = clp**2 * olg / (1 + clp * olg) + chp
    else:
        a_in = clp * (1 + olg) / (1 + clp * olg)
        a_sense = chp * olg / (1 + clp * olg)
        a_disp = chp / (1 + clp * olg)

    return a_in, a_sense, a_disp

#####   Parameters   #####
if __name__ == "__main__":

    st.title("Third-generation frequency noise budgets")
    st.write()

    sb = st.sidebar

    sb.header("Inspiral parameters")


    # Frequency vector
    fflow = 10 # Hz
    ffhigh = 5e6 # Hz
    fflog = np.logspace(np.log10(fflow), np.log10(ffhigh), 1000)

    ### CE input mode cleaners parameters
    # Both
    mass = 2.92 # kg
    Pin = 500 # W
    p_in_eff_imc1 = 10 # W
    p_in_imc1 = 25e-3 # W
    p_in_imc2 = 25e-3 # W
    p_in_ce = 25e-3 # W

    mod_depth = np.sqrt(2) # rads
    epsilon = 1e-5

    temperature = 295 # K
    poisson_ratio = 0.17 # -
    youngs_modulus = 71e9 # Pascals
    coating_thickness = 4.5e-6 # m
    loss_angle = 4e-4 # rads

    # IMC1
    length_imc1 = sb.number_input('First IMC length [m]', min_value=10, max_value=4000, value=100) # m
    finesse_imc1 = 700 
    fsr_imc1 = fsr(length_imc1)
    fpole_imc1 = fsr_imc1 / (2 * finesse_imc1)
    beam_radius_imc1 = beam_waist(length_imc1)

    # IMC2
    length_imc2 = sb.number_input('Second IMC length [m]', min_value=10, max_value=4000, value=330) # m # m
    finesse_imc2 = 700 
    fsr_imc2 = fsr(length_imc2)
    fpole_imc2 = fsr_imc2 / (2 * finesse_imc2)
    beam_radius_imc2 = beam_waist(length_imc2)

    # CE
    length_ce = 40e3 # m
    fsr_ce = fsr(length_ce)
    beam_radius_ce = beam_waist(length_ce)

    ### Control loops
    # PSL 
    bandwidth_psl = 500e3 # Hz
    boost_stages_psl = 2
    olg_psl = fss(fflog, boost_stages_psl, bandwidth_psl)

    # IMC1
    bandwidth_imc1 = sb.number_input('First IMC control loop bandwidth [Hz]', min_value=1000, max_value=1000000, value=100000) # Hz
    boost_stages_imc1 = 3
    olg_imc1 = fss(fflog, boost_stages_imc1, bandwidth_imc1)

    RR_imc1 = get_cavity_reflectivity_from_finesse(finesse_imc1)
    clp_imc1 = cavity_low_pass(fflog, RR_imc1, fsr_imc1)
    chp_imc1 = 1 - clp_imc1

    a_in_imc1, a_sense_imc1, a_disp_imc1 = loop_noise_coupling_functions(olg_imc1, clp_imc1, chp_imc1, lock='active')

    # IMC2
    bandwidth_imc2 = sb.number_input('Second IMC control loop bandwidth [Hz]', min_value=1, max_value=1000, value=30) # Hz
    boost_stages_imc2 = 2
    olg_imc2 = fss(fflog, boost_stages_imc2, bandwidth_imc2)

    RR_imc2 = get_cavity_reflectivity_from_finesse(finesse_imc2)
    clp_imc2 = cavity_low_pass(fflog, RR_imc2, fsr_imc2)
    chp_imc2 = 1 - clp_imc2

    a_in_imc2, a_sense_imc2, a_disp_imc2 = loop_noise_coupling_functions(olg_imc2, clp_imc2, chp_imc2, lock='passive')

    # CE 
    bandwidth_ce = 200 # Hz
    boost_stages_ce = 2
    olg_ce = fss(fflog, boost_stages_ce, bandwidth_ce)

    Tp = 0.03
    Ti = 0.014
    Te = 4e-6
    rp = np.sqrt(1 - Tp)
    ri = np.sqrt(1 - Ti)
    re = np.sqrt(1 - Te)
    lp = 200 # m
    L_ce = 40e3 # m
    fpole_ce = coupled_cavity_pole(rp, ri, re, L_ce)
    finesse_ce = fsr_ce / (2 * fpole_ce)
    clp_ce = coupled_cavity_low_pass(fflog, rp, ri, re, lp, L_ce)
    chp_ce = 1 - clp_ce

    a_in_ce, a_sense_ce, a_disp_ce = loop_noise_coupling_functions(olg_ce, clp_ce, chp_ce, lock='active')


    ### CE requirement
    ce_freq_noise_mag_req = 7e-7 # Hz/rtHz, assumes our limit is dominated by HOMs
    ce_darm_pole = 825 # Hz
    ce_requirement = requirement_ce_freq_noise(fflog, ce_freq_noise_mag_req, ce_darm_pole)


    ### CE IMC sensing noise
    sensing_noise_raw_imc1_incl_reflmc = sensing_noise(fflog, p_in_eff_imc1, finesse_imc1, fsr_imc1, fpole_imc1)
    sensing_noise_raw_imc1 = sensing_noise(fflog, p_in_imc1, finesse_imc1, fsr_imc1, fpole_imc1)
    sensing_noise_raw_imc2 = sensing_noise(fflog, p_in_imc2, finesse_imc2, fsr_imc2, fpole_imc2)
    sensing_noise_raw_ce = sensing_noise(fflog, p_in_ce, finesse_ce, fsr_ce, fpole_ce)

    sensing_noise_imc1_incl_reflmc = sensing_noise_raw_imc1_incl_reflmc * np.abs(a_sense_imc1)
    sensing_noise_imc1 = sensing_noise_raw_imc1 * np.abs(a_sense_imc1)
    sensing_noise_imc2 = sensing_noise_raw_imc2 * np.abs(a_sense_imc2)
    sensing_noise_ce = sensing_noise_raw_ce * np.abs(a_sense_ce)

    ### CE IMC displacement noise
    # radiation pressure noise
    rad_press_raw_imc1 = radiation_pressure_freq_noise(fflog, Pin, length_imc1, finesse_imc1)
    rad_press_raw_imc2 = radiation_pressure_freq_noise(fflog, Pin, length_imc2, finesse_imc2)
    rad_press_raw_ce = radiation_pressure_freq_noise(fflog, Pin, length_ce, finesse_ce, mass=320)
    rad_press_imc1 = rad_press_raw_imc1 * np.abs(a_disp_imc1)
    rad_press_imc2 = rad_press_raw_imc2 * np.abs(a_disp_imc2)
    rad_press_ce = rad_press_raw_ce * np.abs(a_disp_ce)

    # coating brownian noise
    coating_noise_raw_imc1 = coating_brownian_freq_noise(fflog, length_imc1, beam_radius_imc1)
    coating_noise_raw_imc2 = coating_brownian_freq_noise(fflog, length_imc2, beam_radius_imc2)
    coating_noise_raw_ce = ce_coating_brownian_freq_noise(fflog)
    coating_noise_imc1 = coating_noise_raw_imc1 * np.abs(a_disp_imc1)
    coating_noise_imc2 = coating_noise_raw_imc2 * np.abs(a_disp_imc2)
    coating_noise_ce = coating_noise_raw_ce * np.abs(a_disp_ce)

    ### CE laser noise
    # NRPO
    npro_freerunning = npro_freq_noise(fflog)
    npro_suppressed = npro_freerunning * np.abs(1 / (1 + olg_psl))
    npro_imc1 = npro_suppressed * np.abs(clp_imc1 / (1 + clp_imc1 * olg_imc1))

    # VCO + Ref cav
    vco_raw_noise = vco_freq_noise(fflog)
    vco_imc1 = vco_raw_noise * np.abs(clp_imc1 / (1 + clp_imc1 * olg_imc1))

    ### Apply Input coupling for IMC2 to IMC1 output frequency noises
    npro_in_imc2 = npro_imc1 * np.abs(a_in_imc2)
    vco_in_imc2 = vco_imc1 * np.abs(a_in_imc2)
    sensing_noise_in_imc2_incl_reflmc = sensing_noise_imc1_incl_reflmc * np.abs(a_in_imc2)
    sensing_noise_in_imc2 = sensing_noise_imc1 * np.abs(a_in_imc2)
    rad_press_in_imc2 = rad_press_imc1 * np.abs(a_in_imc2)
    coating_noise_in_imc2 = coating_noise_imc1 * np.abs(a_in_imc2)

    ### Apply Input coupling for CE to IMC2 output frequency noises
    npro_in_imc2_ce = npro_in_imc2 * np.abs(a_in_ce)
    vco_in_imc2_ce = vco_in_imc2 * np.abs(a_in_ce)
    sensing_noise_in_imc2_incl_reflmc_ce = sensing_noise_in_imc2_incl_reflmc * np.abs(a_in_ce)
    sensing_noise_in_imc2_ce = sensing_noise_in_imc2 * np.abs(a_in_ce)
    rad_press_in_imc2_ce = rad_press_in_imc2 * np.abs(a_in_ce)
    coating_noise_in_imc2_ce = coating_noise_in_imc2 * np.abs(a_in_ce)

    sensing_noise_imc2_in_ce = sensing_noise_imc2 * np.abs(a_in_ce)
    rad_press_imc2_in_ce = rad_press_imc2 * np.abs(a_in_ce)
    coating_noise_imc2_in_ce = coating_noise_imc2 * np.abs(a_in_ce)

    # print statements
    if False:
        print()
        print(f"fpole_imc1 = {fpole_imc1} Hz")
        print(f"fpole_imc2 = {fpole_imc2} Hz")
        print(f"fpole_ce   = {fpole_ce} Hz")
        print()
        print(f"fsr_imc1 = {fsr_imc1} Hz")
        print(f"fsr_imc2 = {fsr_imc2} Hz")
        print(f"fsr_ce   = {fsr_ce} Hz")
        print()
        print(f"finesse_imc1 = {finesse_imc1}")
        print(f"finesse_imc2 = {finesse_imc2}")
        print(f"finesse_ce   = {finesse_ce}")

    # st.write(f'First IMC length = {length_imc1:.0f} m')
    # st.write(f'Second IMC length = {length_imc2:.0f} m')
    # st.write(f'First IMC bandwidth = {bandwidth_imc1:.0f} Hz')
    # st.write(f'Second IMC bandwidth = {bandwidth_imc2:.0f} Hz')

    st.latex(f"\mathrm{{First\ IMC\ length:}} \qquad " + r"L_{IMC1}" + f" = \mathbf{{{length_imc1:.0f}}}\ " + r"\mathrm{m}")
    st.latex(f"\mathrm{{Second\ IMC\ length:}} \qquad " + r"L_{IMC2}" + f" = \mathbf{{{length_imc2:.0f}}}\ " + r"\mathrm{m}")
    st.latex(f"\mathrm{{First\ IMC\ bandwidth:}} \qquad " + r"BW_{IMC1}" + f" = \mathbf{{{bandwidth_imc1:.0f}}}\ " + r"\mathrm{Hz}")
    st.latex(f"\mathrm{{Second\ IMC\ bandwidth:}} \qquad " + r"BW_{IMC2}" + f" = \mathbf{{{bandwidth_imc2:.0f}}}\ " + r"\mathrm{Hz}")

    ##### Figures #####
    plot_names = np.array([])
    yylow = 1e-8 # Hz/rtHz
    yyhigh = 1e-3 # Hz/rtHz

    # ### Frequency stabilization servos
    # fig, (s1, s2) = plt.subplots(2, sharex=True)

    # s1.loglog(fflog, np.abs(clp_imc1), label='IMC1')
    # s1.loglog(fflog, np.abs(clp_imc2), label='IMC2')
    # s1.loglog(fflog, np.abs(clp_ce), label='CE')

    # s2.semilogx(fflog, np.angle(clp_imc1, deg=True), label='IMC1')
    # s2.semilogx(fflog, np.angle(clp_imc2, deg=True), label='IMC2')
    # s2.semilogx(fflog, np.angle(clp_ce, deg=True), label='CE')

    # s1.set_xlim([fflog[0], fflog[-1]])
    # # s1.set_ylim([10.0**-5, 10**12])
    # s2.set_yticks([-180, -90, 0, 90, 180])
    # # s1.set_yticks(10.0**np.arange(-5, 13))
    # # s1.set_ylim([yylow, yyhigh])

    # s1.set_title('Cavity low pass filters')
    # s1.set_ylabel('Mag [Hz/Hz]')
    # s2.set_ylabel('Phase [deg]')
    # s2.set_xlabel('Frequency '+r'$f$ [Hz]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)
    # s2.grid()
    # s2.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'third_gen_cavity_low_pass_filters.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()
    

    # ### Frequency stabilization servos
    # fig, (s1, s2) = plt.subplots(2, sharex=True)

    # s1.loglog(fflog, np.abs(olg_imc1), label='IMC1')
    # s1.loglog(fflog, np.abs(olg_imc2), label='IMC2')
    # s1.loglog(fflog, np.abs(olg_ce), label='CE')

    # s2.semilogx(fflog, np.angle(olg_imc1, deg=True), label='IMC1')
    # s2.semilogx(fflog, np.angle(olg_imc2, deg=True), label='IMC2')
    # s2.semilogx(fflog, np.angle(olg_ce, deg=True), label='CE')

    # s1.set_xlim([fflog[0], fflog[-1]])
    # s1.set_ylim([10.0**-5, 10**12])
    # s2.set_yticks([-180, -90, 0, 90, 180])
    # s1.set_yticks(10.0**np.arange(-5, 13))
    # # s1.set_ylim([yylow, yyhigh])

    # s1.set_title('Control loop open loop gains')
    # s1.set_ylabel('OLG Mag')
    # s2.set_ylabel('Phase [deg]')
    # s2.set_xlabel('Frequency '+r'$f$ [Hz]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)
    # s2.grid()
    # s2.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'third_gen_olgs.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()


    # ### Frequency coupling functions IMC1
    # fig, (s1, s2) = plt.subplots(2, sharex=True)

    # s1.loglog(fflog, np.abs(a_in_imc1), label='Input')
    # s1.loglog(fflog, np.abs(a_sense_imc1), label='Sense')
    # s1.loglog(fflog, np.abs(a_disp_imc1), label='Displacement')

    # s2.semilogx(fflog, np.angle(a_in_imc1, deg=True), label='Input')
    # s2.semilogx(fflog, np.angle(a_sense_imc1, deg=True), label='Sense')
    # s2.semilogx(fflog, np.angle(a_disp_imc1, deg=True), label='Displacement')

    # s1.set_xlim([fflog[0], fflog[-1]])
    # s1.set_ylim([10.0**-3, 3])
    # s2.set_yticks([-180, -90, 0, 90, 180])
    # # s1.set_yticks(10.0**np.arange(-5, 13))
    # # s1.set_ylim([yylow, yyhigh])

    # s1.set_title('Frequency noise coupling to trans IMC1')
    # s1.set_ylabel('Mag')
    # s2.set_ylabel('Phase [deg]')
    # s2.set_xlabel('Frequency '+r'$f$ [Hz]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)
    # s2.grid()
    # s2.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'freq_noise_coupling_to_trans_imc1.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()


    # ### Frequency coupling functions IMC2
    # fig, (s1, s2) = plt.subplots(2, sharex=True)

    # s1.loglog(fflog, np.abs(a_in_imc2), label='Input')
    # s1.loglog(fflog, np.abs(a_sense_imc2), label='Sense')
    # s1.loglog(fflog, np.abs(a_disp_imc2), label='Displacement')

    # s2.semilogx(fflog, np.angle(a_in_imc2, deg=True), label='Input')
    # s2.semilogx(fflog, np.angle(a_sense_imc2, deg=True), label='Sense')
    # s2.semilogx(fflog, np.angle(a_disp_imc2, deg=True), label='Displacement')

    # s1.set_xlim([fflog[0], fflog[-1]])
    # s1.set_ylim([10.0**-3, 3])
    # s2.set_yticks([-180, -90, 0, 90, 180])
    # # s1.set_yticks(10.0**np.arange(-5, 13))
    # # s1.set_ylim([yylow, yyhigh])

    # s1.set_title('Frequency noise coupling to trans IMC2')
    # s1.set_ylabel('Mag')
    # s2.set_ylabel('Phase [deg]')
    # s2.set_xlabel('Frequency '+r'$f$ [Hz]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)
    # s2.grid()
    # s2.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'freq_noise_coupling_to_trans_imc2.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()


    # ### Frequency coupling functions CE
    # fig, (s1, s2) = plt.subplots(2, sharex=True)

    # s1.loglog(fflog, np.abs(a_in_ce), label='Input')
    # s1.loglog(fflog, np.abs(a_sense_ce), label='Sense')
    # s1.loglog(fflog, np.abs(a_disp_ce), label='Displacement')

    # s2.semilogx(fflog, np.angle(a_in_ce, deg=True), label='Input')
    # s2.semilogx(fflog, np.angle(a_sense_ce, deg=True), label='Sense')
    # s2.semilogx(fflog, np.angle(a_disp_ce, deg=True), label='Displacement')

    # s1.set_xlim([fflog[0], fflog[-1]])
    # s1.set_ylim([10.0**-3, 3])
    # s2.set_yticks([-180, -90, 0, 90, 180])
    # # s1.set_yticks(10.0**np.arange(-5, 13))
    # # s1.set_ylim([yylow, yyhigh])

    # s1.set_title('Frequency noise coupling to trans CE')
    # s1.set_ylabel('Mag')
    # s2.set_ylabel('Phase [deg]')
    # s2.set_xlabel('Frequency '+r'[Hz]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)
    # s2.grid()
    # s2.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'freq_noise_coupling_to_trans_ce.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()

    #########################################   Main plots   #########################################
    ### Single input mode cleaner frequency budget
    fig, (s1) = plt.subplots(1, figsize=(12,7.5))

    s1.loglog(fflog, np.abs(ce_requirement), lw=4, color='C4', label="CE requirement")

    s1.loglog(fflog, np.abs(sensing_noise_imc1_incl_reflmc), color='C0', label='Sensing (with refl MC)')
    s1.loglog(fflog, np.abs(sensing_noise_imc1), color='C0', alpha=0.5, label='Sensing (no refl MC)')
    s1.loglog(fflog, np.abs(rad_press_imc1), color='C2', label='Radiation pressure')
    s1.loglog(fflog, np.abs(coating_noise_imc1), color='C3', label='Coating brownian')
    s1.loglog(fflog, np.abs(npro_imc1), color='C1', label='NPRO + Ref Cav')
    s1.loglog(fflog, np.abs(vco_imc1), color='C5', label='VCO')

    
    s1.set_xlim([fflog[0], fflog[-1]])
    s1.set_ylim([yylow, yyhigh])

    s1.set_xlabel('Frequency '+r'[Hz]')
    s1.set_ylabel('Frequency noise ' + r'[$\mathrm{Hz}/\sqrt{\mathrm{Hz}}$]')

    s1.legend(fontsize=18)

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'ce_freq_noise_budget_single_imc.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()

    st.header("Frequency noise budget - One input mode cleaner")
    st.write(fig)


    ### Double input mode cleaner frequency budget
    fig, (s1) = plt.subplots(1, figsize=(12,7.5))

    s1.loglog(fflog, np.abs(ce_requirement), lw=4, color='C4', label="CE requirement")

    s1.loglog(fflog, np.abs(sensing_noise_imc2), color='C0', ls='--', label='Sensing IMC2')
    s1.loglog(fflog, np.abs(rad_press_imc2), color='C2', ls='--', label='Radiation pressure IMC2')
    s1.loglog(fflog, np.abs(coating_noise_imc2), color='C3', ls='--', label='Coating brownian IMC2')

    s1.loglog(fflog, np.abs(sensing_noise_in_imc2_incl_reflmc), color='C0', label='Sensing IMC1 (with refl MC)')
    s1.loglog(fflog, np.abs(sensing_noise_in_imc2), color='C0', alpha=0.5, label='Sensing IMC1 (no refl MC)')
    s1.loglog(fflog, np.abs(rad_press_in_imc2), color='C2', label='Radiation pressure IMC1')
    s1.loglog(fflog, np.abs(coating_noise_in_imc2), color='C3', label='Coating brownian IMC1')
    s1.loglog(fflog, np.abs(npro_in_imc2), color='C1', label='NPRO + Ref Cav')
    s1.loglog(fflog, np.abs(vco_in_imc2), color='C5', label='VCO')


    s1.set_xlim([fflog[0], fflog[-1]])
    s1.set_ylim([yylow, yyhigh])

    s1.set_xlabel('Frequency '+r'[Hz]')
    s1.set_ylabel('Frequency noise ' + r'[$\mathrm{Hz}/\sqrt{\mathrm{Hz}}$]')

    s1.legend(fontsize=18, ncol=2)

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'ce_freq_noise_budget_double_imc.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()

    st.header("Frequency noise budget - Two input mode cleaners")
    st.write(fig)


    # ### Double input mode cleaner plus CE EFL CARM frequency budget
    # fig, (s1) = plt.subplots(1)

    # s1.loglog(fflog, np.abs(ce_requirement), lw=4, color='C4', label="CE requirement")

    # s1.loglog(fflog, np.abs(sensing_noise_ce), color='C0', ls=':', label='Sensing CE')
    # s1.loglog(fflog, np.abs(rad_press_ce), color='C2', ls=':', label='Radiation pressure CE')
    # # s1.loglog(fflog, np.abs(coating_noise_ce), color='C3', ls=':', label='Coating brownian CE')

    # s1.loglog(fflog, np.abs(sensing_noise_imc2_in_ce), color='C0', ls='--', label='Sensing IMC2')
    # s1.loglog(fflog, np.abs(rad_press_imc2_in_ce), color='C2', ls='--', label='Radiation pressure IMC2')
    # s1.loglog(fflog, np.abs(coating_noise_imc2_in_ce), color='C3', ls='--', label='Coating brownian IMC2')

    # s1.loglog(fflog, np.abs(sensing_noise_in_imc2_incl_reflmc_ce), color='C0', label='Sensing IMC1')
    # s1.loglog(fflog, np.abs(rad_press_in_imc2_ce), color='C2', label='Radiation pressure IMC1')
    # s1.loglog(fflog, np.abs(coating_noise_in_imc2_ce), color='C3', label='Coating brownian IMC1')
    # s1.loglog(fflog, np.abs(npro_in_imc2_ce), color='C1', label='NPRO + Ref Cav')
    # s1.loglog(fflog, np.abs(vco_in_imc2_ce), color='C5', label='VCO')


    # s1.set_xlim([fflog[0], fflog[-1]])
    # s1.set_ylim([1e-10, 1e-5])

    # s1.set_xlabel('Frequency '+r'[Hz]')
    # s1.set_ylabel('Frequency noise ' + r'[$\mathrm{Hz}/\sqrt{\mathrm{Hz}}$]')

    # s1.legend()

    # s1.grid()
    # s1.grid(which='minor', ls='--', alpha=0.7)

    # plot_name = f'ce_freq_noise_budget_double_imc_plus_ce_refl_carm.pdf'
    # full_plot_name = os.path.join(fig_dir, plot_name)
    # plot_names = np.append(plot_names, full_plot_name)
    # print(f'Writing plot PDF to {full_plot_name}')
    # plt.savefig(full_plot_name, bbox_inches='tight')
    # plt.close()

    # print open command
    # command = 'gopen'
    # for pf in plot_names:
    #     command = '{} {}'.format(command, pf)
    # print()
    # print('Command to open plots generated by this script:')
    # print(command)
    # print()

    st.markdown(f"""
        The code for this web script is at [streamlit_3gfn]({PROJECT_URL}).  
        Please [report any issues]({PROJECT_URL}/-/issues).
    """)

