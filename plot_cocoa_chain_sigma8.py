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

#parameter ranges
ranges={'logAs':(1.61,3.91),
        'ns':(0.87,1.07),
        'H0':(55,91),
        'omegab':(0.03,0.07),
        'omegam':(0.1,0.9)
}

# debug file to plot samples used for training in a triangle plot

base_path = '../cocoa2/Cocoa/projects/lsst_y1/'
names = ['sigma8','ns','H0','omegab','omegam','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5']
label = ['\sigma_8','n_s','H_0','\Omega_b','\Omega_m','\Delta z_S^1','\Delta z_S^2','\Delta z_S^3','\Delta z_S^4','\Delta z_S^5','\mathrm{IA}_1','\mathrm{IA}_2','m^1','m^2','m^3','m^4','m^5']
idxs = [26,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

chain1 = np.loadtxt(base_path+'chains/lsst_at_planck_kmax5.1.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain2 = np.loadtxt(base_path+'chains/lsst_at_planck_kmax5.2.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain3 = np.loadtxt(base_path+'chains/lsst_at_planck_kmax5.3.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain4 = np.loadtxt(base_path+'chains/lsst_at_planck_kmax5.4.txt')[:,idxs]#,4,5,6]]#[22,27]]

len1 = len(chain1)
len2 = len(chain2)
len3 = len(chain3)
len4 = len(chain4)

#burn in and thin
thin_frac = 10
burn_in_frac = 0.5

chain1 = chain1[int(0.5*len1)::thin_frac]
chain2 = chain2[int(0.5*len2)::thin_frac]
chain3 = chain3[int(0.5*len3)::thin_frac]
chain4 = chain4[int(0.5*len4)::thin_frac]

chain = np.vstack((chain1,chain2,chain3,chain4))
print(chain.shape)
mcmc1  = getdist.mcsamples.MCSamples(samples=chain,names=names,labels=label,label='Cocoa',ranges=ranges)
#omm,sigma8,As = compute_omegam_sigma8(mcmc1.samples)

#mcmc1.addDerived(omm,name='omegam',label='\Omega_m')
#mcmc1.addDerived(sigma8,name='sigma8',label='\sigma_8')
#mcmc1.addDerived(As,name='As',label='A_s')

# open and convert emulator chain
samples = np.load('../cocoa2/Cocoa/params_with_sigma8.npy')
names = ['logAs','ns','H0','omegab','omegam','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','sigma8']
label = ['\logA_s','n_s','H_0','\Omega_b','\Omega_m','\Delta z_S^1','\Delta z_S^2','\Delta z_S^3','\Delta z_S^4','\Delta z_S^5','\mathrm{IA}_1','\mathrm{IA}_2','\sigma_8']
chain2 = getdist.mcsamples.MCSamples(samples=samples,names=names,labels=label,label='Emulator', ranges=ranges)


g = plots.get_subplot_plotter()
#GET DIST PLOT SETUP
g = plots.get_subplot_plotter(analysis_settings=analysissettings3,width_inch=8)
g.settings.axis_tick_x_rotation=75
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 14.0
g.settings.legend_fontsize = 12.75
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=20
g.legend_labels=False

param_3d = None

g.settings.num_plot_contours = 2
g.triangle_plot([mcmc1,chain2],
                plot_3d_with_param=param_3d,
                line_args=[
                        {'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
                        {'lw': 1.2,'ls': 'dashed', 'color':'maroon'},
                ],
                contour_colors=['lightcoral','maroon'],
                contour_ls=['solid','dashed'], 
                contour_lws=[1.0,2.0],
                filled=[True,False],
                params=['sigma8','omegam','omegab','H0','ns','IA1','IA2'])#'DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
g.export('plots/cocoa_vs_emu_posteriors_sigma8.pdf')

# ### Compute gaussian tension:
# print(mcmc1.getParamNames().getRunningNames())
# print(chain2.getParamNames().getRunningNames())

# cov_1 = mcmc1.cov()[:17,:17]
# cov_2 = chain2.cov()[:17,:17]

# print(cov_1.shape)

# mean_1 = mcmc1.getMeans()[:17]
# mean_2 = chain2.getMeans()[:17]

# gauss_tension = np.transpose(mean_1-mean_2) @ np.linalg.inv(cov_1+cov_2) @ (mean_1-mean_2)
# print(gauss_tension)






