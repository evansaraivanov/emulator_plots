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
        
    return (omm,sigma_8,As,omb)

#parameter ranges
ranges={'logAs':  (1.61,3.91),
        'ns':     (0.87,1.07),
        'H0':     (55,91),
        'Omegab': (0.03,0.07),
        'omegab': (0.01,0.04),
        'Omegam': (0.1,0.9),
        'omegac': (0.01,0.99)
}

# debug file to plot samples used for training in a triangle plot

base_path = './projects/lsst_y1/emulator_output/'
names = ['logA','ns','H0','omegab','omegac','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2']#,'M1','M2','M3','M4','M5']
label = ['\log A_s','n_s','H_0','\Omega_bh^2','\Omega_ch^2','\Delta z_S^1','\Delta z_S^2','\Delta z_S^3','\Delta z_S^4','\Delta z_S^5','\mathrm{IA}_1','\mathrm{IA}_2']

base_path = '../cocoa2/Cocoa/projects/lsst_y1/emulator_output/chains/'
chain1 = np.load(base_path+'train_t128_samples_0.npy')
chain2 = np.load(base_path+'train_t256_samples_0.npy')
chain3 = np.load(base_path+'train_t512_samples_0.npy')

mcmc1  = getdist.mcsamples.MCSamples(samples=chain1,names=names,labels=label,label='T=128',ranges=ranges)
mcmc2  = getdist.mcsamples.MCSamples(samples=chain2,names=names,labels=label,label='T=256',ranges=ranges)
mcmc3  = getdist.mcsamples.MCSamples(samples=chain3,names=names,labels=label,label='T=512',ranges=ranges)

omm,sigma8,As,omb = compute_omegam_sigma8(mcmc1.samples)

# mcmc1.addDerived(omm,name='Omegam',label='\Omega_m')
# mcmc1.addDerived(omb,name='Omegab',label='\Omega_b')
# mcmc1.addDerived(sigma8,name='sigma8',label='\sigma_8')
# mcmc1.addDerived(As,name='As',label='A_s')

omm,sigma8,As,omb = compute_omegam_sigma8(mcmc2.samples)

# mcmc2.addDerived(omm,name='Omegam',label='\Omega_m')
# mcmc2.addDerived(omb,name='Omegab',label='\Omega_b')
# mcmc2.addDerived(sigma8,name='sigma8',label='\sigma_8')
# mcmc2.addDerived(As,name='As',label='A_s')

omm,sigma8,As,omb = compute_omegam_sigma8(mcmc3.samples)

# mcmc3.addDerived(omm,name='Omegam',label='\Omega_m')
# mcmc3.addDerived(omb,name='Omegab',label='\Omega_b')
# mcmc3.addDerived(sigma8,name='sigma8',label='\sigma_8')
# mcmc3.addDerived(As,name='As',label='A_s')

#GET DIST PLOT SETUP
g = plots.get_subplot_plotter(analysis_settings=analysissettings3,width_inch=15)
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
g.triangle_plot([mcmc1,mcmc2,mcmc3],
                #params=['logA','omegab','omegac','H0','ns'],
                plot_3d_with_param=param_3d,
                line_args=[
                        {'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
                        {'lw': 1.2,'ls': 'dashed', 'color':'maroon'},
                        {'lw': 2.0,'ls': 'solid', 'color':'indigo'}
                ],
                contour_colors=['lightcoral','maroon','indigo'],
                contour_ls=['solid','dashed','solid'], 
                contour_lws=[1.0,2.0,1.2],
                filled=[True,False,False],
                shaded=False,
                legend_loc=(0.35, 0.82)
            )#,'DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
g.export('plots/training_samples.pdf')

g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.triangle_plot([mcmc1,mcmc2,mcmc3],
                #params=['As','Omegab','Omegam','H0','ns'],
                plot_3d_with_param=param_3d,
                line_args=[
                        {'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
                        {'lw': 1.2,'ls': 'dashed', 'color':'maroon'},
                        {'lw': 2.0,'ls': 'solid', 'color':'indigo'}
                ],
                contour_colors=['lightcoral','maroon','indigo'],
                contour_ls=['solid','dashed','solid'], 
                contour_lws=[1.0,2.0,1.2],
                filled=[True,False,False],
                shaded=False,
                legend_loc=(0.35, 0.82)
            )#,'DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
g.export('plots/training_samples_wl.pdf')