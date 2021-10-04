"""Reproduces Figure 4 with IMCnoise.nb mathematica notebook in python.

Craig Cahillane
June 10, 2021
"""

import os
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as scc

mpl.rcParams.update({   'figure.figsize': (12, 9),
                        'text.usetex': True,
                        'font.family': 'serif',
                        # 'font.serif': 'Georgia',
                        # 'mathtext.fontset': 'cm',
                        'lines.linewidth': 5,
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

#####   Set up figures directory   #####
script_path = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(os.path.abspath(__file__)).replace('.py', '')
git_dir = os.path.abspath(os.path.join(script_path, '..'))
fig_dir = f'{git_dir}/figures/{script_name}'
print()
print('Figures made by this script will be placed in:')
print(fig_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)



#####   Functions   #####
def fsr(length):
    """Free spectral range, returned in Hz. Assumes two mirror cavity length in meters"""
    return scc.c/(2 * length)

def cavity_round_trip_losses_approx(length, A0=1e-5, exponent=1/3):
    """Approximation of cavity round-trip losses given the cavity length, and some initial parameters.
    Inputs:
    -------
    length: float
        length of the cavity
    A0: float
        losses for a 1 meter cavity. Default is 10 ppm.
    exponent: float
        exponent of the length factor. Default is 1/3.
    """
    return A0 * length**exponent

def cavity_total_losses(finesse, round_trip_losses):
    """Total cavity losses from round trip losses"""
    return round_trip_losses * finesse / np.pi

def cavity_round_trip_losses(finesse, total_losses):
    """Round-trip cavity losses from total cavity losses"""
    return total_losses * np.pi / finesse

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
    """Circulating power in the cavity"""
    return Pin * finesse / np.pi 

def radiation_pressure_noise(
    ff,
    Pin,
    length,
    finesse,
    mass=2.92,
    lam=1064e-9,
):
    """Radiation pressure noise in units of Hz/rtHz"""
    fsr0 = fsr(length)
    Pcav = circulating_power(Pin, finesse) 
    length_noise_rad_pressure = 2 * np.sqrt(2 * scc.c * scc.h * Pcav / lam) / (scc.c * mass * ff**2)
    freq_noise_rad_pressure = 4 * fsr0 * length_noise_rad_pressure / lam
    return freq_noise_rad_pressure

def beam_waist(length, lam=1064e-9):
    """Minimum beam waist given the cavity length"""
    return np.sqrt(length * lam / np.pi)

def coating_brownian(
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
    """Coating brownian noise in Hz/rtHz"""
    kB = scc.Boltzmann
    foo = 4 * kB * temperature / (np.pi**2 * ff)
    bar = (1 + poisson_ratio) * (1 - 2 * poisson_ratio) / young_modulus
    baz = coating_thickness * loss_angle / beam_radius**2
    Scoat = foo * bar * baz

    fsr0 = fsr(length)
    Fcoat = 2 * fsr0 * np.sqrt(2 * Scoat) / lam
    return Fcoat

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

if __name__ == "__main__":
    L = 16.48998 # m
    R = 0.9939317
    T1 = 6030e-6 # from https://galaxy.ligo.caltech.edu/optics/
    T2 = 5.1e-6
    T3 = 5845e-6
    r1 = np.sqrt(1 - T1)
    r2 = np.sqrt(1 - T2)
    r3 = np.sqrt(1 - T3)
    Art = 40e-6
    freq = 1000 # Hz
    Pin = 500 # W

    FSR = fsr(L)
    FF = finesse(T1, T3, Art)
    pole = cavity_pole(L, T1, T3, Art)
    total_loss = cavity_total_losses(FF, Art)

    print()
    print(f"FSR          = {FSR:.0f} Hz")
    print(f"finesse      = {FF:.0f}")
    print(f"pole         = {pole:.1f} Hz")
    print(f"total_loss   = {100*total_loss:.2f} %")
    print()

    low_length = 10 # m
    high_length = 1000 # m
    lengths = np.logspace(np.log10(low_length), np.log10(high_length))

    total_losses = np.array([
        1,
        3,
        10
    ]) * 1e-2
    lss = np.array([
        '-',
        '--',
        ':',
    ])
    colors = np.array([
        'C0',
        'C1',
        'C2',
    ])

    # Calculate finesse and cavity pole given total cavity losses
    cavity_dict = {}
    for total_loss, color, ls in zip(total_losses, colors, lss):
        temp_round_trip_losses = cavity_round_trip_losses_approx(lengths)
        temp_finesse = np.pi * total_loss / temp_round_trip_losses
        temp_fsr = fsr(lengths)
        temp_cavity_pole = temp_fsr / (2 * temp_finesse)
        temp_rad_pressure = radiation_pressure_noise(freq, Pin, lengths, temp_finesse)
        temp_power_density = max_power_density(Pin, lengths, temp_finesse)

        cavity_dict[total_loss] = {}
        cavity_dict[total_loss]['color'] = color
        cavity_dict[total_loss]['ls'] = ls
        cavity_dict[total_loss]['finesse'] = temp_finesse
        cavity_dict[total_loss]['pole'] = temp_cavity_pole
        cavity_dict[total_loss]['radiation_pressure'] = temp_rad_pressure
        cavity_dict[total_loss]['power_density'] = temp_power_density

    ww = beam_waist(lengths) # m
    plot_coating_brownian = coating_brownian(freq, lengths, ww) # Hz/rtHz

    low_freq_noise_limit = requirement_ce_freq_noise(freq) # Hz/rtHz
    high_freq_noise_limit = 1e-5 # Hz/rtHz

    #####   Figures   #####
    plot_names = np.array([])

    # Plot the cavity finesse
    fig, (s1) = plt.subplots(1)

    for total_loss in total_losses:
        plot_finesses = cavity_dict[total_loss]['finesse']
        color = cavity_dict[total_loss]['color']
        ls = cavity_dict[total_loss]['ls']

        s1.loglog(lengths, plot_finesses, color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")

    s1.set_xlim([lengths[0], lengths[-1]])

    s1.set_xlabel('Cavity length '+r'$L$ [m]')
    s1.set_ylabel('Cavity Finesse ' + r'$\mathcal{F}$')

    s1.legend(title="Total cavity losses " + r"$\delta_\mathrm{tot}(L)$")

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    plot_name = f'cavity_length_vs_cavity_finesse.pdf'
    full_plot_name = os.path.join(fig_dir, plot_name)
    plot_names = np.append(plot_names, full_plot_name)
    print(f'Writing plot PDF to {full_plot_name}')
    plt.savefig(full_plot_name, bbox_inches='tight')
    plt.close()



    # Plot the cavity poles
    fig, (s1) = plt.subplots(1)

    for total_loss in total_losses:
        plot_poles = cavity_dict[total_loss]['pole']
        color = cavity_dict[total_loss]['color']
        ls = cavity_dict[total_loss]['ls']

        s1.loglog(lengths, plot_poles, color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")

    s1.set_xlim([lengths[0], lengths[-1]])

    s1.set_xlabel('Cavity length '+r'$L$ [m]')
    s1.set_ylabel('Cavity pole ' + r'$f_\mathrm{pole}$ [Hz]')

    s1.legend(title="Total cavity losses " + r"$\delta_\mathrm{tot}(L)$")

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    plot_name = f'cavity_length_vs_cavity_poles.pdf'
    full_plot_name = os.path.join(fig_dir, plot_name)
    plot_names = np.append(plot_names, full_plot_name)
    print(f'Writing plot PDF to {full_plot_name}')
    plt.savefig(full_plot_name, bbox_inches='tight')
    plt.close()



    # Plot the radiation pressure noise
    fig, (s1) = plt.subplots(1)

    for total_loss in total_losses:
        plot_rad_pressure = cavity_dict[total_loss]['radiation_pressure']
        color = cavity_dict[total_loss]['color']
        ls = cavity_dict[total_loss]['ls']

        s1.loglog(lengths, plot_rad_pressure, color=color, ls=ls, label=r"$\delta_\mathrm{tot}(L) = $" + f" {100 * total_loss:.0f} \%")

    s1.loglog(lengths, plot_coating_brownian, 'C3', label=f"Coating brownian "  + r"$F_\mathrm{coating}(f=1~\mathrm{kHz})$")
    s1.axhspan(low_freq_noise_limit, high_freq_noise_limit, alpha=0.1, color='blue', 
                    label=f'CE frequency noise req: ' + r'$3 \times 10^{-7}~\mathrm{Hz}/\sqrt{\mathrm{Hz}}$')

    s1.set_xlim([lengths[0], lengths[-1]])
    s1.set_ylim([1e-9, high_freq_noise_limit])

    s1.set_xlabel('Cavity length '+r'$L$ [m]')
    s1.set_ylabel(f'Frequency noise at {freq*1e-3:.0f} kHz ' + r'[$\mathrm{Hz}/\sqrt{\mathrm{Hz}}$]')

    s1.legend(title="Radiation pressure " + r"$F_\mathrm{rad}(f=1~\mathrm{kHz}, P_\mathrm{in} = 500~\mathrm{W})$")

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    plot_name = f'cavity_length_vs_radiation_pressure_noise.pdf'
    full_plot_name = os.path.join(fig_dir, plot_name)
    plot_names = np.append(plot_names, full_plot_name)
    print(f'Writing plot PDF to {full_plot_name}')
    plt.savefig(full_plot_name, bbox_inches='tight')
    plt.close()



    # Plot the cavity power density
    fig, (s1) = plt.subplots(1)

    for total_loss in total_losses:
        plot_power_density = cavity_dict[total_loss]['power_density']
        color = cavity_dict[total_loss]['color']
        ls = cavity_dict[total_loss]['ls']

        s1.loglog(lengths, plot_power_density, color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")

    s1.set_xlim([lengths[0], lengths[-1]])
    s1.set_ylim([1e4, 1e8])

    s1.set_xlabel('Cavity length '+r'$L$ [m]')
    s1.set_ylabel('Power density ' + r'$P_\mathrm{cav}/\pi w_0^2$ [$\mathrm{W}/\mathrm{cm}^2$]')

    s1.legend(title="Total cavity losses " + r"$\delta_\mathrm{tot}(L)$")

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.7)

    plot_name = f'cavity_length_vs_power_density.pdf'
    full_plot_name = os.path.join(fig_dir, plot_name)
    plot_names = np.append(plot_names, full_plot_name)
    print(f'Writing plot PDF to {full_plot_name}')
    plt.savefig(full_plot_name, bbox_inches='tight')
    plt.close()



    # Plot all plots together
    fig, (s1, s2, s3, s4) = plt.subplots(4, sharex=True, figsize=(12,17))

    for total_loss in total_losses:
        plot_finesses = cavity_dict[total_loss]['finesse']
        plot_poles = cavity_dict[total_loss]['pole']
        plot_power_density = cavity_dict[total_loss]['power_density']
        plot_rad_pressure = cavity_dict[total_loss]['radiation_pressure']
        color = cavity_dict[total_loss]['color']
        ls = cavity_dict[total_loss]['ls']

        s1.loglog(lengths, plot_finesses,       color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")
        s2.loglog(lengths, plot_poles,          color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")
        s3.loglog(lengths, plot_power_density,  color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")
        s4.loglog(lengths, plot_rad_pressure,   color=color, ls=ls, label=f"{100 * total_loss:.0f} \%")

    s4.loglog(lengths, plot_coating_brownian, 'C3', label=f"Coating brownian "  + r"$F_\mathrm{coating}$")
    s4.axhspan(low_freq_noise_limit, high_freq_noise_limit, alpha=0.1, color='blue', 
                    label=f'CE frequency noise req') # + r'$3 \times 10^{-7}~\mathrm{Hz}/\sqrt{\mathrm{Hz}}$')

    s4.set_xlim([lengths[0], lengths[-1]])
    s3.set_ylim([1e4, 1e8])
    s4.set_ylim([1e-9, 1e-5])

    s3.set_yticks([1e4, 1e5, 1e6, 1e7, 1e8])
    s4.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])

    s4.set_xlabel('Cavity length '+r'$L$ [m]')

    s1.set_ylabel('Cavity Finesse ' + r'$\mathcal{F}$')
    s2.set_ylabel('Cavity pole ' + r'$f_\mathrm{pole}$ [Hz]')
    s3.set_ylabel('Power density ' + r'$\frac{P_\mathrm{cav}}{\pi w_0^2}$ [$\frac{\mathrm{W}}{\mathrm{cm}^2}$]')
    s4.set_ylabel(f'Frequency noise ' + r'[$\frac{\mathrm{Hz}}{\sqrt{\mathrm{Hz}}}$]')

    s1.legend(title="Total cavity losses " + r"$\delta_\mathrm{tot}(L)$")
    s4.legend(title="Radiation pressure " + r"$F_\mathrm{rad}(f=1~\mathrm{kHz})$", ncol=2)

    s1.grid()
    s1.grid(which='minor', ls='--', alpha=0.6)
    s2.grid()
    s2.grid(which='minor', ls='--', alpha=0.6)
    s3.grid()
    s3.grid(which='minor', ls='--', alpha=0.6)
    s4.grid()
    s4.grid(which='minor', ls='--', alpha=0.6)

    plot_name = f'all_cavity_lengths.pdf'
    full_plot_name = os.path.join(fig_dir, plot_name)
    plot_names = np.append(plot_names, full_plot_name)
    print(f'Writing plot PDF to {full_plot_name}')
    plt.savefig(full_plot_name, bbox_inches='tight')
    plt.close()

    # print open command
    command = 'gopen'
    for pf in plot_names:
        command = '{} {}'.format(command, pf)
    print()
    print('Command to open plots generated by this script:')
    print(command)
    print()