import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import os

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

# root directory of chi2 data
root_path = './delta_chi2_data/'
model_list = ['','_resnet','_mlp','_resbottle'] # empty for transformer, else '_mlp' or '_resnet'
fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=False, squeeze=True, figsize=(15,5))

for i,model in enumerate(model_list):
	# load each
	chi2_t128_600k = np.load(root_path+'t128_600k'+model+'_deltachi2.npy')
	chi2_t128_1200k = np.load(root_path+'t128_1200k'+model+'_deltachi2.npy')
	chi2_t128_3000k = np.load(root_path+'t128_3000k'+model+'_deltachi2.npy')

	chi2_t256_600k = np.load(root_path+'t256_600k'+model+'_deltachi2.npy')
	chi2_t256_1200k = np.load(root_path+'t256_1200k'+model+'_deltachi2.npy')
	chi2_t256_3000k = np.load(root_path+'t256_3000k'+model+'_deltachi2.npy')

	chi2_t512_600k = np.load(root_path+'t512_600k'+model+'_deltachi2.npy')
	chi2_t512_1200k = np.load(root_path+'t512_1200k'+model+'_deltachi2.npy')
	chi2_t512_3000k = np.load(root_path+'t512_3000k'+model+'_deltachi2.npy')
	#print(np.min(chi2_t512_600k),np.median(chi2_t512_600k),np.max(chi2_t512_600k))

	# subplots
	minimum = 1e-2
	maximum = 1e2
	ranges = (minimum,maximum)
	bins1 = np.logspace(np.log10(minimum),np.log10(maximum), 75)
	bins2 = np.logspace(np.log10(minimum),np.log10(maximum), 70)
	bins3 = np.logspace(np.log10(minimum),np.log10(maximum), 70)

	ax[0,i].hist(chi2_t128_3000k, bins=bins1, density=False, range=ranges, facecolor='#2ca02ca0',label='3000k',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[0,i].hist(chi2_t128_1200k, bins=bins2, density=False, range=ranges, facecolor='#ff7f0ea0',label='1200k',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[0,i].hist(chi2_t128_600k,  bins=bins3, density=False, range=ranges, facecolor='#1f77b4a0',label='600k',histtype='step',hatch='|',fill=True,edgecolor='k')

	ax[1,i].hist(chi2_t256_3000k, bins=bins1, density=False, range=ranges, facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[1,i].hist(chi2_t256_1200k, bins=bins2, density=False, range=ranges, facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[1,i].hist(chi2_t256_600k,  bins=bins3, density=False, range=ranges, facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

	ax[2,i].hist(chi2_t512_3000k, bins=bins1, density=False, range=ranges, facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[2,i].hist(chi2_t512_1200k, bins=bins2, density=False, range=ranges, facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[2,i].hist(chi2_t512_600k,  bins=bins3, density=False, range=ranges, facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

#ax.yaxis.set_tick_params(labelleft=False)
for i in range(3):
	for j in range(4):
		ax[i,j].set_yticks([])
		ax[i,j].set_xlim(minimum+1e-5,maximum-1e-5)

# ax[0,0].set_ylabel('Train T=128')
# ax[1,0].set_ylabel('Train T=256')
# ax[2,0].set_ylabel('Train T=512')
# ax[2,0].set_xlabel('Transformer')
# ax[2,1].set_xlabel('ResNet')
# ax[2,2].set_xlabel('MLP')
# ax[2,3].set_xlabel('ResBottle')

#### FOR OUTSIDE TEXT
fig.text(0.2, 0.00, 'RN+TF', ha='center')
fig.text(0.4, 0.00, 'ResNet', ha='center')
fig.text(0.6, 0.00, 'MLP', ha='center')
fig.text(0.8, 0.00, 'ResBottle', ha='center')

fig.text(0.08, 0.25, 'Temp = 512', va='center', rotation='vertical')
fig.text(0.08, 0.5, 'Temp = 256', va='center', rotation='vertical')
fig.text(0.08, 0.75, 'Temp = 128', va='center', rotation='vertical')

fig.text(0.515, 0.04, '$\chi^2$', ha='center',fontsize=18.0)
fig.text(0.1, 0.5, 'Count', va='center', rotation='vertical',fontsize=18.0)
####

####  FOR INSIDE  TEXT
# fig.text(0.13, 0.32, 'Transformer', ha='left')
# fig.text(0.325, 0.32, 'ResNet', ha='left')
# fig.text(0.525, 0.32, 'MLP', ha='left')
# fig.text(0.725, 0.32, 'ResBottle', ha='left')

# fig.text(0.725, 0.79, 'Temp = 512', ha='left')
# fig.text(0.725, 0.54, 'Temp = 256', ha='left')
# fig.text(0.725, 0.29, 'Temp = 128', ha='left')

# fig.text(0.515, 0.04, '$\chi^2$', ha='center',fontsize=18.0)
# fig.text(0.1, 0.5, 'Count', va='center', rotation='vertical',fontsize=18.0)
####

plt.xscale("log")
plt.subplots_adjust(hspace=.1,wspace=.1)
ax[0,0].legend()
fig.savefig('plots/chi2_test_all.pdf')
