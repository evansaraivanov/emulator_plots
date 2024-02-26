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
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16

# root directory of chi2 data
root_path = './delta_chi2_data/'
root_path_new = './delta_chi2_new/'
model_list = ['_tanh_optimal','_resnet']#,'_mlp','_resbottle'] # empty for transformer, else '_mlp' or '_resnet'
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, squeeze=True, figsize=(15,5))

bin_count = np.array([
				[65,65,65, 70,70,75, 65,70,75],
				[70,70,70, 100,70,65, 70,70,65]
			])

for i,model in enumerate(model_list):
	# load each
	if model=='_tanh_optimal':
		sfx = '.txt'
		chi2_t128_600k  = np.loadtxt(root_path_new+'T128_600k'+model+sfx)
		chi2_t128_1200k = np.loadtxt(root_path_new+'T128_1200k'+model+sfx)
		chi2_t128_3000k = np.loadtxt(root_path_new+'T128_3000k'+model+sfx)

		chi2_t256_600k  = np.loadtxt(root_path_new+'T256_600k'+model+sfx)
		chi2_t256_1200k = np.loadtxt(root_path_new+'T256_1200k'+model+sfx)
		chi2_t256_3000k = np.loadtxt(root_path_new+'T256_3000k'+model+sfx)

		chi2_t512_600k  = np.loadtxt(root_path_new+'T512_600k'+model+sfx)
		chi2_t512_1200k = np.loadtxt(root_path_new+'T512_1200k'+model+sfx)
		chi2_t512_3000k = np.loadtxt(root_path_new+'T512_3000k'+model+sfx)
	else:
		sfx = '.npy'
		chi2_t128_600k  = np.load(root_path+'t128_600k'+model+'_deltachi2'+sfx)
		chi2_t128_1200k = np.load(root_path+'t128_1200k'+model+'_deltachi2'+sfx)
		chi2_t128_3000k = np.load(root_path+'t128_3000k'+model+'_deltachi2'+sfx)

		chi2_t256_600k  = np.load(root_path+'t256_600k'+model+'_deltachi2'+sfx)
		chi2_t256_1200k = np.load(root_path+'t256_1200k'+model+'_deltachi2'+sfx)
		chi2_t256_3000k = np.load(root_path+'t256_3000k'+model+'_deltachi2'+sfx)

		chi2_t512_600k  = np.load(root_path+'t512_600k'+model+'_deltachi2'+sfx)
		chi2_t512_1200k = np.load(root_path+'t512_1200k'+model+'_deltachi2'+sfx)
		chi2_t512_3000k = np.load(root_path+'t512_3000k'+model+'_deltachi2'+sfx)
	#print(np.min(chi2_t512_600k),np.median(chi2_t512_600k),np.max(chi2_t512_600k))

	# get counts
	print(model,':')

	print('chi2_t128_600k',len(np.where(chi2_t128_600k>1)[0]),len(np.where(chi2_t128_600k>0.2)[0]),len(chi2_t128_600k))
	print('chi2_t128_1200k',len(np.where(chi2_t128_1200k>1)[0]),len(np.where(chi2_t128_1200k>0.2)[0]),len(chi2_t128_1200k))
	print('chi2_t128_3000k',len(np.where(chi2_t128_3000k>1)[0]),len(np.where(chi2_t128_3000k>0.2)[0]),len(chi2_t128_3000k))

	print('chi2_t256_600k',len(np.where(chi2_t256_600k>1)[0]),len(np.where(chi2_t256_600k>0.2)[0]),len(chi2_t256_600k))
	print('chi2_t256_1200k',len(np.where(chi2_t256_1200k>1)[0]),len(np.where(chi2_t256_1200k>0.2)[0]),len(chi2_t256_1200k))
	print('chi2_t256_3000k',len(np.where(chi2_t256_3000k>1)[0]),len(np.where(chi2_t256_3000k>0.2)[0]),len(chi2_t256_3000k))

	print('chi2_t512_600k',len(np.where(chi2_t512_600k>1)[0]),len(np.where(chi2_t512_600k>0.2)[0]),len(chi2_t512_600k))
	print('chi2_t512_1200k',len(np.where(chi2_t512_1200k>1)[0]),len(np.where(chi2_t512_1200k>0.2)[0]),len(chi2_t512_1200k))
	print('chi2_t512_3000k',len(np.where(chi2_t512_3000k>1)[0]),len(np.where(chi2_t512_3000k>0.2)[0]),len(chi2_t512_3000k))

	# subplots
	minimum = 1e-2
	maximum = 1e2
	ranges = (minimum,maximum)

	bins1 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,0])
	bins2 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,1])
	bins3 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,2])

	bins4 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,3])
	bins5 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,4])
	bins6 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,5])

	bins7 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,6])
	bins8 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,7])
	bins9 = np.logspace(np.log10(minimum),np.log10(maximum), bin_count[i,8])

	ax[i,0].hist(chi2_t128_3000k, bins=bins1, density=False, range=ranges, facecolor='#2ca02ca0',label='3000k',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[i,0].hist(chi2_t128_1200k, bins=bins2, density=False, range=ranges, facecolor='#ff7f0ea0',label='1200k',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[i,0].hist(chi2_t128_600k,  bins=bins3, density=False, range=ranges, facecolor='#1f77b4a0',label='600k',histtype='step',hatch='|',fill=True,edgecolor='k')

	ax[i,1].hist(chi2_t256_3000k, bins=bins4, density=False, range=ranges, facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[i,1].hist(chi2_t256_1200k, bins=bins5, density=False, range=ranges, facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[i,1].hist(chi2_t256_600k,  bins=bins6, density=False, range=ranges, facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

	ax[i,2].hist(chi2_t512_3000k, bins=bins7, density=False, range=ranges, facecolor='#2ca02ca0',histtype='step',hatch='-',fill=True,edgecolor='k')
	ax[i,2].hist(chi2_t512_1200k, bins=bins8, density=False, range=ranges, facecolor='#ff7f0ea0',histtype='step',hatch='/',fill=True,edgecolor='k')
	ax[i,2].hist(chi2_t512_600k,  bins=bins9, density=False, range=ranges, facecolor='#1f77b4a0',histtype='step',hatch='|',fill=True,edgecolor='k')

#ax.yaxis.set_tick_params(labelleft=False)
for i in range(2):
	for j in range(3):
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
fig.text(0.115, 0.7, 'ResTRF', va='center', ha='center', rotation='vertical', fontsize=16.0)
fig.text(0.115, 0.29, 'ResMLP', va='center', ha='center', rotation='vertical', fontsize=16.0)
# fig.text(0.6, 0.00, 'MLP', ha='center')
# fig.text(0.8, 0.00, 'ResBottle', ha='center')

fig.text(0.25, 0.92, 'T = 128', va='center', ha='center', fontsize=16.0)
fig.text(0.512, 0.92, 'T = 256', va='center', ha='center', fontsize=16.0)
fig.text(0.78, 0.92, 'T = 512', va='center', ha='center', fontsize=16.0)

fig.text(0.515, 0.02, '$\Delta\chi^2$', va='center', ha='center', fontsize=22.0)
fig.text(0.1, 0.5, 'Count', va='center', ha='center', rotation='vertical', fontsize=22.0)
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
plt.subplots_adjust(hspace=.0,wspace=.0)
ax[0,0].legend()
fig.savefig('plots/chi2_test_rn_tf.pdf')
