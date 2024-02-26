import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os
import scipy
import sys

#===== Plot settings =============================================
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
matplotlib.rcParams['axes.labelsize'] = 'large'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['legend.fontsize'] = 14

x_multiplier = 0.02
y_multiplier = 0.9

#===== Setup the figure =================================
fig, ax = plt.subplots(figsize=(5, 4))
#===== Normalizing flow =================================
shift0_nf  = np.loadtxt('../tension_calibration_update/scripts/nf_0.txt')
shiftp5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_p5.txt')
shiftm5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_m5.txt')

bins_count = 15

ymin = 1e-5
ymax = 1.3

xmin = 0
xmax = 3.4

bins = np.linspace(xmin,xmax,bins_count)

# we can regularize the binning by eliminating data outside the plot
idxs1 = np.where(shift0_nf>=xmin)
shift0_nf = shift0_nf[idxs1]
idxs2 = np.where(shift0_nf<=xmax)
shift0_nf = shift0_nf[idxs2]

idxs1 = np.where(shiftp5_nf>=xmin)
shiftp5_nf = shiftp5_nf[idxs1]
idxs2 = np.where(shiftp5_nf<=xmax)
shiftp5_nf = shiftp5_nf[idxs2]

idxs1 = np.where(shiftm5_nf>=xmin)
shiftm5_nf = shiftm5_nf[idxs1]
idxs2 = np.where(shiftm5_nf<=xmax)
shiftm5_nf = shiftm5_nf[idxs2]

ax.hist(shift0_nf,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax.hist(shiftp5_nf,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax.hist(shiftm5_nf,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax.tick_params(which='minor', length=4)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
xmin = 0
xmax = 3.4

ax.set_ylabel('Count')
#ax[0,0].set_xlabel(r'$N_\sigma$')
ax.set_xlabel(r'$N_\sigma$')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
#ax[0,0].legend()
ax.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),'Param Difference')
plt.savefig('plots/nf.pdf')

#===== Setup the figure =================================
fig, ax = plt.subplots(figsize=(5, 4))
#===== Eigentension =================================
shift0_eig  = np.loadtxt('../tension_calibration_update/scripts/eig_0.txt')
shiftp5_eig = np.loadtxt('../tension_calibration_update/scripts/eig_p5.txt')
shiftm5_eig = np.loadtxt('../tension_calibration_update/scripts/eig_m5.txt')

bins_count = 18

ymin = 1e-5
ymax = 2.3

xmin = 0
xmax = 2.7

bins = np.linspace(xmin,xmax,bins_count)

# we can regularize the binning by eliminating data outside the plot
idxs1 = np.where(shift0_eig>=xmin)
shift0_eig = shift0_eig[idxs1]
idxs2 = np.where(shift0_eig<=xmax)
shift0_eig = shift0_eig[idxs2]

idxs1 = np.where(shiftp5_eig>=xmin)
shiftp5_eig = shiftp5_eig[idxs1]
idxs2 = np.where(shiftp5_eig<=xmax)
shiftp5_eig = shiftp5_eig[idxs2]

idxs1 = np.where(shiftm5_eig>=xmin)
shiftm5_eig = shiftm5_eig[idxs1]
idxs2 = np.where(shiftm5_eig<=xmax)
shiftm5_eig = shiftm5_eig[idxs2]

ax.hist(shift0_eig,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax.hist(shiftp5_eig,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax.hist(shiftm5_eig,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax.tick_params(which='minor', length=4)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
xmin = 0
xmax = 3.4

ax.set_ylabel('Count')
#ax[0,1].set_ylabel('Count')
#ax[0,13_xlabel(r'$N_\sigma$')
ax.set_xlabel(r'$N_\sigma$')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

ax.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),'Eigentension')
plt.savefig('plots/eig.pdf')


#===== Setup the figure =================================
fig, ax = plt.subplots(figsize=(5, 4))
#===== QUDM update form =================================
shift0_qudm  = np.loadtxt('../tension_calibration_update/scripts/qudm_0.txt')
shiftp5_qudm = np.loadtxt('../tension_calibration_update/scripts/qudm_p5.txt')
shiftm5_qudm = np.loadtxt('../tension_calibration_update/scripts/qudm_m5.txt')

bins_count = 15

ymin = 1e-5
ymax = 1.2

xmin = 1e-20
xmax = 3.4

bins = np.linspace(xmin,xmax,bins_count)

# we can regularize the binning by eliminating data outside the plot
idxs1 = np.where(shift0_qudm>=xmin)
shift0_qudm = shift0_qudm[idxs1]
idxs2 = np.where(shift0_qudm<=xmax)
shift0_qudm = shift0_qudm[idxs2]

idxs1 = np.where(shiftp5_qudm>=xmin)
shiftp5_qudm = shiftp5_qudm[idxs1]
idxs2 = np.where(shiftp5_qudm<=xmax)
shiftp5_qudm = shiftp5_qudm[idxs2]

idxs1 = np.where(shiftm5_qudm>=xmin)
shiftm5_qudm = shiftm5_qudm[idxs1]
idxs2 = np.where(shiftm5_qudm<=xmax)
shiftm5_qudm = shiftm5_qudm[idxs2]

ax.hist(shift0_qudm,density=True,bins=bins,label='Cosmology 1',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax.hist(shiftp5_qudm,density=True,bins=bins,label='Cosmology 4',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax.hist(shiftm5_qudm,density=True,bins=bins,label='Cosmology 5',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax.tick_params(which='minor', length=4)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
xmin = 0
xmax = 3.4

ax.set_ylabel('Count')
ax.set_xlabel(r'$N_\sigma$')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
# ax[1,0].legend()
#ax.legend(bbox_to_anchor=(0.35, -0.25), loc='upper left')
ax.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'$Q_\mathrm{UDM}$')
plt.savefig('plots/qudm.pdf')


#===== Setup the figure =================================
fig, ax = plt.subplots(figsize=(5, 4))
#===== QDMAP GOF degradation =================================
shift0_qdmap  = np.loadtxt('../tension_calibration_update/scripts/qdmap_0.txt')
shiftp5_qdmap = np.loadtxt('../tension_calibration_update/scripts/qdmap_p5.txt')
shiftm5_qdmap = np.loadtxt('../tension_calibration_update/scripts/qdmap_m5.txt')

bins_count = 18

ymin = 1e-5
ymax = 1.4

xmin = 0
xmax = 3.4

bins = np.linspace(xmin,xmax,bins_count)

# we can regularize the binning by eliminating data outside the plot
idxs1 = np.where(shift0_qdmap>=xmin)
shift0_qdmap = shift0_qdmap[idxs1]
idxs2 = np.where(shift0_qdmap<=xmax)
shift0_qdmap = shift0_qdmap[idxs2]

idxs1 = np.where(shiftp5_qdmap>=xmin)
shiftp5_qdmap = shiftp5_qdmap[idxs1]
idxs2 = np.where(shiftp5_qdmap<=xmax)
shiftp5_qdmap = shiftp5_qdmap[idxs2]

idxs1 = np.where(shiftm5_qdmap>=xmin)
shiftm5_qdmap = shiftm5_qdmap[idxs1]
idxs2 = np.where(shiftm5_qdmap<=xmax)
shiftm5_qdmap = shiftm5_qdmap[idxs2]

ax.hist(shift0_qdmap,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax.hist(shiftp5_qdmap,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax.hist(shiftm5_qdmap,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax.tick_params(which='minor', length=4)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
xmin = 0
xmax = 3.4

#ax[1,1].set_ylabel('Count')
ax.set_ylabel('Count')
ax.set_xlabel(r'$N_\sigma$')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
# ax[1,0].legend()

ax.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'$Q_\mathrm{DMAP}$')
plt.savefig('plots/qdmap.pdf')

#===== Setup the figure =================================
fig, ax = plt.subplots(figsize=(5, 4))
#===== Suspiciousness =================================
shift0_qdmap  = np.loadtxt('../tension_calibration_update/scripts/sus_0.txt')
shiftp5_qdmap = np.loadtxt('../tension_calibration_update/scripts/sus_p5.txt')
shiftm5_qdmap = np.loadtxt('../tension_calibration_update/scripts/sus_m5.txt')

bins_count = 18

ymin = 1e-5
ymax = 1.4

xmin = 0
xmax = 3.4

bins = np.linspace(xmin,xmax,bins_count)

# we can regularize the binning by eliminating data outside the plot
idxs1 = np.where(shift0_qdmap>=xmin)
shift0_qdmap = shift0_qdmap[idxs1]
idxs2 = np.where(shift0_qdmap<=xmax)
shift0_qdmap = shift0_qdmap[idxs2]

idxs1 = np.where(shiftp5_qdmap>=xmin)
shiftp5_qdmap = shiftp5_qdmap[idxs1]
idxs2 = np.where(shiftp5_qdmap<=xmax)
shiftp5_qdmap = shiftp5_qdmap[idxs2]

idxs1 = np.where(shiftm5_qdmap>=xmin)
shiftm5_qdmap = shiftm5_qdmap[idxs1]
idxs2 = np.where(shiftm5_qdmap<=xmax)
shiftm5_qdmap = shiftm5_qdmap[idxs2]

ax.hist(shift0_qdmap,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax.hist(shiftp5_qdmap,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax.hist(shiftm5_qdmap,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax.tick_params(which='minor', length=4)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
xmin = 0
xmax = 3.4

#ax[1,1].set_ylabel('Count')
ax.set_ylabel('Count')
ax.set_xlabel(r'$N_\sigma$')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
# ax[1,0].legend()

ax.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'Suspiciousness')


#save and go
#plt.tight_layout()
plt.savefig('plots/sus.pdf')

