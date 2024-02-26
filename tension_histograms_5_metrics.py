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
fig = plt.figure(figsize=(10, 12))

gs = gridspec.GridSpec(5, 1, figure=fig)
gs.update(left=0.05, right=0.48, wspace=1, hspace=0.05)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[2, 0])
ax4 = plt.subplot(gs[3, 0])
ax5 = plt.subplot(gs[4, 0])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)

# take difference?
difference = 1 
p0_nf = np.loadtxt('../tension_calibration_update/scripts/nf_0.txt')
p5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_p5.txt')
m5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_m5.txt')

#===== Normalizing flow =================================
shift0_nf  = np.loadtxt('../tension_calibration_update/scripts/nf_0.txt') - difference*p0_nf
shiftp5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_p5.txt') - difference*p5_nf
shiftm5_nf = np.loadtxt('../tension_calibration_update/scripts/nf_m5.txt') - difference*m5_nf

bins_count = 15

ymin = 1e-5
ymax = 1.3

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

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

ax1.hist(shift0_nf,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax1.hist(shiftp5_nf,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax1.hist(shiftm5_nf,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax1.tick_params(which='minor', length=2)
ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

#ax1.set_ylabel('Count')
#ax[0,0].set_xlabel(r'$N_\sigma$')
#ax1.set_xlabel(r'$N_\sigma$')
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)
#ax[0,0].legend()
ax1.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),'Param Difference')



#===== Eigentension =================================
shift0_eig  = np.loadtxt('../tension_calibration_update/scripts/eig_0.txt') - difference*p0_nf
shiftp5_eig = np.loadtxt('../tension_calibration_update/scripts/eig_p5.txt') - difference*p5_nf
shiftm5_eig = np.loadtxt('../tension_calibration_update/scripts/eig_m5.txt') - difference*m5_nf

bins_count = 18

ymin = 1e-5
ymax = 2.3

xmin = 0
xmax = 2.7

if difference:
	xmin = -2
	xmax = 2

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

ax2.hist(shift0_eig, density=True,bins=bins,facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k',label='Cosmo 1')
ax2.hist(shiftp5_eig,density=True,bins=bins,facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')#,label='Cosmology 4')
ax2.hist(shiftm5_eig,density=True,bins=bins,facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')#,label='Cosmology 5')

ax2.tick_params(which='minor', length=2)
ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

xmin = 0
xmax = 3.2

if difference:
	xmin = -2
	xmax = 2

#ax[0,13_xlabel(r'$N_\sigma$')
#ax2.set_xlabel(r'$N_\sigma$')
ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin,ymax)

ax2.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),'Eigentension')



#===== QUDM update form =================================
shift0_qudm  = np.loadtxt('../tension_calibration/scripts/qudm_0.txt') - difference*p0_nf
shiftp5_qudm = np.loadtxt('../tension_calibration/scripts/qudm_p5.txt') - difference*p5_nf
shiftm5_qudm = np.loadtxt('../tension_calibration/scripts/qudm_m5.txt') - difference*m5_nf

bins_count = 20

ymin = 1e-5
ymax = 1.6

xmin = 1e-20
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

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

ax3.hist(shift0_qudm, density=True,bins=bins,facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')#,label='Cosmology 1')
ax3.hist(shiftp5_qudm,density=True,bins=bins,facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k',label='Cosmo 4')
ax3.hist(shiftm5_qudm,density=True,bins=bins,facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')#,label='Cosmology 5')

ax3.tick_params(which='minor', length=2)
ax3.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax3.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

xmin = 0
xmax = 3.4
if difference:
	xmin = -2
	xmax = 2

ax3.set_ylabel('Count')
#ax3.set_xlabel(r'$N_\sigma$')
ax3.set_xlim(xmin,xmax)
ax3.set_ylim(ymin,ymax)
ax3.legend()
#ax3.legend(bbox_to_anchor=(0.35, -0.25), loc='upper left')
ax3.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'$Q_\mathrm{UDM}$')



#===== QDMAP GOF degradation =================================
shift0_qdmap  = np.loadtxt('../tension_calibration_update/scripts/qdmap_0.txt') - difference*p0_nf
shiftp5_qdmap = np.loadtxt('../tension_calibration_update/scripts/qdmap_p5.txt') - difference*p5_nf
shiftm5_qdmap = np.loadtxt('../tension_calibration_update/scripts/qdmap_m5.txt') - difference*m5_nf

bins_count = 18

ymin = 1e-5
ymax = 1.4

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

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

ax4.hist(shift0_qdmap, density=True,bins=bins,facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')#,label='Cosmology 1')
ax4.hist(shiftp5_qdmap,density=True,bins=bins,facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')#,label='Cosmology 4')
ax4.hist(shiftm5_qdmap,density=True,bins=bins,facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k',label='Cosmo 5')

ax4.tick_params(which='minor', length=2)
ax4.legend()
ax4.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax4.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

#ax[1,1].set_ylabel('Count')
#ax4.set_ylabel('Count')
#ax4.set_xlabel(r'$N_\sigma$')
ax4.set_xlim(xmin,xmax)
ax4.set_ylim(ymin,ymax)
# ax[1,0].legend()

ax4.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'$Q_\mathrm{DMAP}$')



#===== Suspiciousness =================================
shift0_qdmap  = np.loadtxt('../tension_calibration_update/scripts/sus_0.txt') - difference*p0_nf
shiftp5_qdmap = np.loadtxt('../tension_calibration_update/scripts/sus_p5.txt') - difference*p5_nf
shiftm5_qdmap = np.loadtxt('../tension_calibration_update/scripts/sus_m5.txt') - difference*m5_nf

bins_count = 18

ymin = 1e-5
ymax = 1.4

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

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

ax5.hist(shift0_qdmap,density=True,bins=bins,label=r'$0\sigma_{PC}$',facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
ax5.hist(shiftp5_qdmap,density=True,bins=bins,label=r'$+5\sigma_{PC}$',facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
ax5.hist(shiftm5_qdmap,density=True,bins=bins,label=r'$-5\sigma_{PC}$',facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

ax5.tick_params(which='minor', length=2)
ax5.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax5.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

xmin = 0
xmax = 3.4

if difference:
	xmin = -2
	xmax = 2

#ax[1,1].set_ylabel('Count')
ax5.set_xlabel(r'$N_\sigma$')
ax5.set_xlim(xmin,xmax)
ax5.set_ylim(ymin,ymax)
# ax[1,0].legend()

ax5.text(x_multiplier*(xmax-xmin),y_multiplier*(ymax-ymin),r'Suspiciousness')

ax2.legend()
#save and go
#plt.tight_layout()
plt.savefig('plots/tension_histograms.pdf')

