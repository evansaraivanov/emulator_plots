##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torchinfo import summary

# just for convinience
from datetime import datetime

def compute_omegam_sigma8(theta):
    logAs = theta[:,0]
    ns    = theta[:,1]
    H0    = theta[:,2]
    ombh2 = theta[:,3]
    omch2 = theta[:,4]
    
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    
    h = H0/100
    As = np.exp(logAs)/(10**10)
    
    omb = ombh2/(h**2)
    omc = omch2/(h**2)
    omn = omnh2/(h**2)
    
    omm = omb+omc+omn
    ommh2 = omm*(h**2)
    
    sigma_8 = (As/3.135e-9)**(1/2) * \
              (ombh2/0.024)**(-0.272) * \
              (ommh2/0.14)**(0.513) * \
              (3.123*h)**((ns-1)/2) * \
              (h/0.72)**(0.698) * \
              (omm/0.27)**(0.236) * \
              (1-0.014)
        
    return (omm,sigma_8,As)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16


# open validation samples
# !!! Watch thin factor !!!
samples_validation_1 = np.load('/home/grads/data/evan/emulator/projects/lsst_y1/emulator_output/chains/valid_post_T64_atplanck_samples_0.npy')
chi2_results_1       = np.loadtxt('./delta_chi2_new/T128_3000k_tanh_optimal.txt')
samples_validation_2 = np.load('/home/grads/data/evan/emulator/projects/lsst_y1/emulator_output/chains/train_t128_samples_1.npy')
chi2_results_2       = np.loadtxt('./delta_chi2_new/T256_3000k_tanh_optimal.txt')
samples_validation_3 = np.load('/home/grads/data/evan/emulator/projects/lsst_y1/emulator_output/chains/train_t256_samples_1.npy')
chi2_results_3       = np.loadtxt('./delta_chi2_new/T512_3000k_tanh_optimal.txt')

# thin
target_n = 10000
thin_factor_1 = len(samples_validation_1)//target_n
thin_factor_2 = len(samples_validation_2)//target_n
thin_factor_3 = len(samples_validation_3)//target_n

if thin_factor_1!=0:
    samples_validation_1 = samples_validation_1[::thin_factor_1]
if thin_factor_2!=0:
    samples_validation_2 = samples_validation_2[::thin_factor_2]
if thin_factor_3!=0:
    samples_validation_3 = samples_validation_3[::thin_factor_3]



fig, ax = plt.subplots(1, 3, figsize=(12, 4),sharex=True,sharey=True)
cmap = plt.cm.get_cmap('coolwarm')

#plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
im0 = ax[0].scatter(samples_validation_1[:,0], samples_validation_1[:,4], c=chi2_results_1, label=r'$T_{\mathrm{test}}=64$ ', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
im1 = ax[1].scatter(samples_validation_2[:,0], samples_validation_2[:,4], c=chi2_results_2, label=r'$T_{\mathrm{test}}=128$', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
im2 = ax[2].scatter(samples_validation_3[:,0], samples_validation_3[:,4], c=chi2_results_3, label=r'$T_{\mathrm{test}}=256$', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
#plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(H0, Omegab, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())

cbar_ax = fig.add_axes([0.3, 0.94, 0.4, 0.05])
fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
plt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, top=0.8)

ax[0].set_xlabel(r'$\log(10^{10}A_s)$')
ax[0].set_ylabel(r'$\Omega_\mathrm{c}h^2$')
ax[1].set_xlabel(r'$\log(10^{10}A_s)$')
ax[2].set_xlabel(r'$\log(10^{10}A_s)$')

ax[0].legend()
ax[1].legend()
ax[2].legend()
cbar_ax.set_xlabel(r'$\Delta\chi^2$',)

ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
ax[2].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

cbar_ax.xaxis.set_label_coords(0.5,-0.5)

plt.savefig("plots/heatmap_omc.pdf")
plt.figure().clear()


### with omegam

omm1,_1,_2 = compute_omegam_sigma8(samples_validation_1)
omm2,_1,_2 = compute_omegam_sigma8(samples_validation_2)
omm3,_1,_2 = compute_omegam_sigma8(samples_validation_3)

fig, ax = plt.subplots(1, 3, figsize=(12, 4),sharex=True,sharey=True)
cmap = plt.cm.get_cmap('coolwarm')

#plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
im0 = ax[0].scatter(samples_validation_1[:,0], omm1, c=chi2_results_1, label=r'$T_{\mathrm{test}}=64$ ', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
im1 = ax[1].scatter(samples_validation_2[:,0], omm2, c=chi2_results_2, label=r'$T_{\mathrm{test}}=128$', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
im2 = ax[2].scatter(samples_validation_3[:,0], omm3, c=chi2_results_3, label=r'$T_{\mathrm{test}}=256$', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e1))
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
#plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(H0, Omegab, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())

cbar_ax = fig.add_axes([0.3, 0.94, 0.4, 0.05])
fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
plt.subplots_adjust(left=0.07, bottom=0.15, right=0.95, top=0.8)

ax[0].set_xlabel(r'$\log(10^{10}A_s)$')
ax[0].set_ylabel(r'$\Omega_\mathrm{m}$')
ax[1].set_xlabel(r'$\log(10^{10}A_s)$')
ax[2].set_xlabel(r'$\log(10^{10}A_s)$')

ax[0].legend()
ax[1].legend()
ax[2].legend()
cbar_ax.set_xlabel(r'$\Delta\chi^2$',)

ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
ax[2].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

cbar_ax.xaxis.set_label_coords(0.5,-0.5)

plt.savefig("plots/heatmap_omm.pdf")




