import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt
import os
import matplotlib

# GENERAL PLOT OPTIONS
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

analysissettings={
  'ignore_rows': u'0.4',
}

analysissettings2={
  'ignore_rows': u'0.4',
}

analysissettings3={
  'smooth_scale_1D':0.3,
  'smooth_scale_2D':0.3,
  'ignore_rows': u'0.0',
  'range_confidence' : u'0.005'
}

# function to convert omega_m and sigma8
def compute_omegam_sigma8(theta):
    logAs = theta[:,0]
    ns    = theta[:,1]
    H0    = theta[:,2]
    omb   = theta[:,3] #ombh2 = theta[:,3]
    omm   = theta[:,4] #omch2 = theta[:,4]
    
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    
    h = H0/100
    As = np.exp(logAs)/(10**10)
    
    ombh2 = omb*(h**2)
    omn   = omnh2/(h**2)
    omc   = (omm - omb - omn)
    omch2 = omc*(h**2)
    ommh2 = omm*(h**2)
    
    sigma_8 = (As/3.135e-9)**(1/2) * \
              (ombh2/0.024)**(-0.272) * \
              (ommh2/0.14)**(0.513) * \
              (3.123*h)**((ns-1)/2) * \
              (h/0.72)**(0.698) * \
              (omm/0.27)**(0.236) * \
              (1-0.014)
        
    return (sigma_8,As,omm)

def compute_omegam_sigma8_emu(theta):
    logAs = theta[:,0]
    ns    = theta[:,1]
    H0    = theta[:,2]
    ombh2 = theta[:,3] #ombh2 = theta[:,3]
    omch2 = theta[:,4] #omch2 = theta[:,4]
    
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    
    h = H0/100
    As = np.exp(logAs)/(10**10)
    
    omb   = ombh2/(h**2)
    omn   = omnh2/(h**2)
    omc   = omch2/(h**2)
    omm   = omb + omc + omn
    ommh2 = omm*(h**2)
    
    sigma_8 = (As/3.135e-9)**(1/2) * \
              (ombh2/0.024)**(-0.272) * \
              (ommh2/0.14)**(0.513) * \
              (3.123*h)**((ns-1)/2) * \
              (h/0.72)**(0.698) * \
              (omm/0.27)**(0.236) * \
              (1-0.014)
        
    return (sigma_8,As,omm)


#parameter ranges
# ranges={'logAs':(1.61,3.91),
#         'ns':(0.87,1.07),
#         'H0':(55,91),
#         'omegab':(0.03,0.07),
#         'omegam':(0.1,0.9)
# }

ranges={
      'Omegam': (0.11,0.64),
      'sigma8': (0.46,1.29)
}

# Open chain

names = ['logA','ns','H0','omegab','omegam','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5']
label = ['\log A_s','n_s','H_0','\Omega_b','\Omega_m','\Delta z_S^1','\Delta z_S^2','\Delta z_S^3','\Delta z_S^4','\Delta z_S^5','\mathrm{IA}_1','\mathrm{IA}_2','m^1','m^2','m^3','m^4','m^5']

samples = np.load('./chains/params_with_sigma8_training.npy')
#print(samples[:,:5])
#print(samples[:,-1])
sigma8_train = samples[:,-1]
sig8,As,omm = compute_omegam_sigma8_emu(samples)
mcmc_train_projected = getdist.mcsamples.MCSamples(samples=np.transpose([sigma8_train,omm]),
                                      names=['sigma8','Omegam'],
                                      labels=['\sigma_8','\Omega_m'],
                                      ranges=ranges,
                                      label='Training samples')

# open and convert emulator chain
#chain = getdist.mcsamples.loadMCSamples('../cocoa2/Cocoa/lsst_at_planck_test_mobo',no_cache=True)
samples_post = np.load('./chains/params_with_sigma8.npy')
#print(samples_post[:,:5])
# print(samples[:,-1])
idxs = [-1,4]
# print(samples[:,idxs])
sigma8 = samples_post[:,-1]
omm = samples_post[:,4]
chain = getdist.mcsamples.MCSamples(samples=np.transpose([sigma8,omm]),
                                      names=['sigma8','Omegam'],
                                      labels=['\sigma_8','\Omega_m'],
                                      ranges=ranges,
                                      label='MCMC chain')
#sigma8,As,omegam = compute_omegam_sigma8(chain.samples)
#omm = chain.samples[:,4]

chain_subspace = chain
means = chain_subspace.getMeans()
cov  = chain_subspace.cov()

mean_sigma8  = means[0]
mean_omm = means[1]
sdev_sigma8  = np.sqrt(cov[0,0])
sdev_omm = np.sqrt(cov[1,1])

normed_sigma8  = (sigma8  - mean_sigma8 )/sdev_sigma8
normed_omm = (omm - mean_omm)/sdev_omm

chain_subspace_norm = getdist.mcsamples.MCSamples(samples=np.transpose(np.array([normed_sigma8,normed_omm])),names=['sigma8','Omegam'])

# Get covariance and shift
cov = chain_subspace_norm.cov()
mean = chain_subspace_norm.getMeans()

eig_vals,eig_vecs = np.linalg.eigh(cov)

idx = np.argmin(eig_vals)

eigval = eig_vals[idx]
eigvec = eig_vecs[idx]

# print(eigval)
# print(eigvec)
# print(eigvec[0]*sdev_sigma8,eigvec[1]*sdev_omm)

shifts = [20,-20]
params = []
for shift in shifts:
  params.append((shift*np.sqrt(eigval)*eigvec+mean)*np.array([sdev_sigma8,sdev_omm]) + means)
  print((shift*np.sqrt(eigval)*eigvec+mean)*np.array([sdev_sigma8,sdev_omm]) + means)
chain_of_shifts = getdist.mcsamples.MCSamples(samples=np.array(params),names=['sigma8','Omegam'],labels=['\sigma_8','\Omega_\mathrm{m}'])

# chain_train = np.load('../cocoa2/Cocoa/projects/lsst_y1/emulator_output/chains/train_t512_samples_0.npy')
# mcmc_train  = getdist.mcsamples.MCSamples(samples=chain_train,names=names[:12],labels=label[:12],label='T=512',ranges=ranges)
# sig8,As,omm = compute_omegam_sigma8_emu(mcmc_train.samples)
# mcmc_train_projected = getdist.mcsamples.MCSamples(samples=np.transpose(np.array([sig8,omm])),names=['sigma8','Omegam'],labels=['\sigma_8','\Omega_\mathrm{m}'],label='Training Points')

# print(sig8,omm)
g = plots.get_subplot_plotter()
#GET DIST PLOT SETUP
g = plots.get_subplot_plotter(analysis_settings=analysissettings3,width_inch=6)
#g.settings.axis_tick_x_rotation=75
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 16.0
g.settings.legend_fontsize = 16
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=20
g.legend_labels=True

g.settings.num_plot_contours = 2
#g.plot_2d_scatter(chain_of_shifts,'sigma8','Omegam',size=2,marker='x',color='r')
g.plot_2d(chain,'sigma8','Omegam',filled=True,contour_colors='lightcoral',line_args=[{'lw': 1.0,'ls': 'solid', 'color':'lightcoral'}],ls='solid',lws=1.0,ranges=ranges)
g.plot_2d(mcmc_train_projected,'sigma8','Omegam',filled=False,line_args=[{'lw':2.0,'ls':'dashed','color':'k'}],ranges=ranges)
plt.scatter(chain_of_shifts.samples[0,0],chain_of_shifts.samples[0,1],marker='x',color='dodgerblue',label='Cosmology 2')
plt.scatter(chain_of_shifts.samples[1,0],chain_of_shifts.samples[1,1],marker='x',color='orange',label='Cosmology 3')
g.add_legend(['T=1 MCMC chain','T=512 training samples'])
# print(np.log(sigma8))

plt.savefig('plots/shifts.pdf')

#####
#
# Con
#
#####





