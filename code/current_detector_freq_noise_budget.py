'''
current_detector_freq_noise_budget.py

Craig Cahillane
Oct 4, 2021
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as scc
import scipy.special as scp
import os


mpl.rcParams.update({'figure.figsize':(12,9),
                     'text.usetex': True,
                     'font.family': 'serif',
                     # 'font.serif': 'Georgia',
                     # 'mathtext.fontset': 'cm',
                     'lines.linewidth': 2.5,
                     'font.size': 22,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'legend.fancybox': True,
                     'legend.fontsize': 18,
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'savefig.dpi': 80,
                     'pdf.compression': 9})

savefigs = True


#####   Set up figures directory   #####
script_path = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.basename(os.path.abspath(__file__)).replace('.py', '')
git_dir = os.path.abspath(os.path.join(script_path, '..'))
fig_dir = f'{git_dir}/figures/{script_name}'
data_dir = f"{git_dir}/data"
print()
print('Figures made by this script will be placed in:')
print(fig_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


### Functions
def save_txt(filename, fff, asd):
    """Save file txt with two columns: frequency vector and power spectral density (ASD)
    """
    save_data = np.vstack((fff, asd)).T
    full_filename = f"{data_dir}/{filename}.txt"
    np.savetxt(full_filename, save_data, header=f"{filename}\nFrequency [Hz], Frequency noise [Hz/rtHz]")
    return

def load_txt(filename):
    """Load file txt with two columns: frequency vector and power spectral density (ASD)
    """
    full_filename = f"{data_dir}/{filename}.txt"
    data = np.loadtxt(full_filename)
    fff = data[:,0]
    asd = data[:,1]
    return fff, asd

### Load data
refl_ctrl_ss_logff, refl_ctrl_ss_logASD = load_txt("incident_frequency_noise")
fflog, cal_IMC_REFL_alone_shot_noise2 = load_txt("input_mode_cleaner_shot_noise")
refl_ss_logff, refl_ss_logASD = load_txt("suppressed_incident_frequency_noise")
fflog, cal_REFL_A_shot_noise2 = load_txt("inteferometer_reflection_shot_noise", )
fflog, cal_IMC_REFL_shot_noise2 = load_txt("suppressed_mode_cleaner_shot_noise", )
aligo_design_equiv_ff, aligo_design_equiv_asd = load_txt("adv_ligo_design_equivalent_noise")
fflog, ce_requirement = load_txt("cosmic_explorer_noise_requirement")


### Figures
fig, (s1) = plt.subplots(1)

# REFL_SERVO_CTRL
s1.loglog(refl_ctrl_ss_logff, refl_ctrl_ss_logASD,
            label='Incident frequency noise')

s1.loglog(fflog, cal_IMC_REFL_alone_shot_noise2,
            label='Input mode cleaner shot noise') # after power increase

# REFL_SERVO_ERR
s1.loglog(refl_ss_logff, refl_ss_logASD,
            label='Suppressed incident frequency noise')

# Shot noise levels
s1.loglog(fflog, cal_REFL_A_shot_noise2,
            label='Interferometer reflection shot noise') # with split sensing

s1.loglog(fflog, cal_IMC_REFL_shot_noise2, 
            label='Suppressed mode cleaner shot noise') # after power increase

# Projections
s1.loglog(aligo_design_equiv_ff, aligo_design_equiv_asd, ls='--',
            label='Adv. LIGO design equivalent noise')

s1.loglog(fflog, ce_requirement, ls='--',
            label='Cosmic Explorer noise requirement')

s1.set_xlim([10, 7000])
s1.set_ylim([1e-9, 1.01e-3])

s1.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
s1.set_xlabel('Frequency [Hz]')
s1.set_ylabel(r'Frequency Noise [Hz/$\sqrt{\mathrm{Hz}}$]')
# s1.set_title('LHO Frequency Noisebudget - April 27, 2019')

s1.grid()
s1.grid(which='minor', ls='--', alpha=0.6)
# s1.legend(loc=(0.01, 0.5), fontsize=16)
s1.legend(fontsize=16)

# Save Plot
if savefigs:
    plot_name = 'frequencyNB.pdf'
    full_plot_name = f'{fig_dir}/{plot_name}'

    print('Writing plot PDF to {}'.format(full_plot_name))
    plt.savefig(full_plot_name, bbox_inches='tight')
