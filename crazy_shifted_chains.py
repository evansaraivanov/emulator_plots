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
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 18

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

base_path = './'

## print(os.listdir('./chains/'))
## print(np.where(np.array(os.listdir('./chains/'))=='chain_emu_20sigma.1.txt'))
## chain6 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_emu_20sigma',no_cache=True)
## print(chain6.samples)

## chain5 = np.loadtxt(base_path+'chains/chain_emu_-20sigma.1.txt')[:,[2,6]]
## chain6 = np.loadtxt(base_path+'chains/chain_emu_20sigma.1.txt')[:,[2,6]]

chain1 = getdist.mcsamples.loadMCSamples(base_path+'chains/lsst_at_planck_kmax5',no_cache=True,settings=analysissettings2)
chain2 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_20sigma',no_cache=True,settings=analysissettings2)
chain3 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_-20sigma',no_cache=True,settings=analysissettings2)

chain4 = getdist.mcsamples.loadMCSamples(base_path+'chains/emulator_chain_T256_3000k_tanh_R1_0.01_0',no_cache=True,settings=analysissettings2)
chain5 = getdist.mcsamples.loadMCSamples(base_path+'chains/emulator_chain_T256_3000k_tanh_R1_0.01_p20',no_cache=True,settings=analysissettings2)
chain6 = getdist.mcsamples.loadMCSamples(base_path+'chains/emulator_chain_T256_3000k_tanh_R1_0.01_m20',no_cache=True,settings=analysissettings2)

chain4.setParamNames(['logA','ns','H0','omegab','omegam','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','w','x','y','z'])
chain5.setParamNames(['logA','ns','H0','omegab','omegam','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','w','x','y','z'])
chain6.setParamNames(['logA','ns','H0','omegab','omegam','dz1','dz2','dz3','dz4','dz5','IA1','IA2','m1','m2','m3','m4','m5','w','x','y','z'])

chain1 = chain1.copy('Cocoa (Cosmo 1)')
chain2 = chain2.copy('Cocoa (Cosmo 2)')
chain3 = chain3.copy('Cocoa (Cosmo 3)')

chain4 = chain4.copy('Emu (Cosmo 1)')
chain5 = chain5.copy('Emu (Cosmo 2)')
chain6 = chain6.copy('Emu (Cosmo 3)')

#GET DIST PLOT SETUP
g = plots.get_subplot_plotter(analysis_settings=analysissettings3,width_inch=6,rc_sizes=True)
g.settings.axis_tick_x_rotation=75
#g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 14.0
g.settings.legend_fontsize = 16
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=20
g.legend_labels=False

param_3d = None

g.settings.num_plot_contours = 2
g.triangle_plot([chain1,chain2,chain3,chain4,chain5,chain6],
                plot_3d_with_param=param_3d,
                line_args=[
                        {'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
                        {'lw': 1.0,'ls': 'solid', 'color':'dodgerblue'},
                        {'lw': 1.0,'ls': 'solid', 'color':'orange'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'maroon'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'darkblue'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'darkorange'},
                ],
                contour_colors=['lightcoral','dodgerblue','orange','maroon','darkblue','darkorange'],
                contour_ls=['solid','solid','solid','dashed','dashed','dashed'], 
                contour_lws=[1.0,1.0,1.0,2.0,2.0,2.0],
                filled=[True,True,True,False,False,False],
                #params=['logA','omegam','omegab','H0','ns'])#'IA1','IA2'])#'DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
                params=['logA','omegam'])

g.export('plots/shifted_chains.pdf')

###### make a plot with sigma8

chain1 = getdist.mcsamples.loadMCSamples(base_path+'chains/lsst_at_planck_kmax5',no_cache=True,settings=analysissettings2)
chain2 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_20sigma',no_cache=True,settings=analysissettings2)
chain3 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_-20sigma',no_cache=True,settings=analysissettings2)

#chain4 = getdist.mcsamples.loadMCSamples(base_path+'chains/lsst_at_planck_test_mobo',no_cache=True,settings=analysissettings2)
#chain5 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_emu_-20sigma',no_cache=True,settings=analysissettings2)
#chain6 = getdist.mcsamples.loadMCSamples(base_path+'chains/chain_emu_20sigma',no_cache=True,settings=analysissettings2)

samples4 = np.load(base_path+'chains/T256_tanh_3000k_R1_0.01_0sigma_with_sigma8.npy')[:,[4,-1]]
samples5 = np.load(base_path+'chains/T256_tanh_3000k_R1_0.01_p20sigma_with_sigma8.npy')[:,[4,-1]]
samples6 = np.load(base_path+'chains/T256_tanh_3000k_R1_0.01_m20sigma_with_sigma8.npy')[:,[4,-1]]

chain4 = getdist.mcsamples.MCSamples(samples=samples4,names=['omegam','sigma8'],labels=['$\Omega_m$','$\sigma_8$'],settings=analysissettings2)
chain5 = getdist.mcsamples.MCSamples(samples=samples5,names=['omegam','sigma8'],labels=['$\Omega_m$','$\sigma_8$'],settings=analysissettings2)
chain6 = getdist.mcsamples.MCSamples(samples=samples6,names=['omegam','sigma8'],labels=['$\Omega_m$','$\sigma_8$'],settings=analysissettings2)

chain1 = chain1.copy('Cocoa (Cosmo 1)')
chain2 = chain2.copy('Cocoa (Cosmo 2)')
chain3 = chain3.copy('Cocoa (Cosmo 3)')

chain4 = chain4.copy('Emu (Cosmo 1)')
chain5 = chain5.copy('Emu (Cosmo 2)')
chain6 = chain6.copy('Emu (Cosmo 3)')

#GET DIST PLOT SETUP
g = plots.get_subplot_plotter(analysis_settings=analysissettings3,width_inch=6,rc_sizes=True)
#g.settings.axis_tick_x_rotation=75
#g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 16.0
g.settings.legend_fontsize = 16
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=20
g.legend_labels=False

param_3d = None

g.settings.num_plot_contours = 2
g.triangle_plot([chain1,chain2,chain3,chain4,chain5,chain6],
                plot_3d_with_param=param_3d,
                line_args=[
                        {'lw': 1.0,'ls': 'solid', 'color':'lightcoral'},
                        {'lw': 1.0,'ls': 'solid', 'color':'dodgerblue'},
                        {'lw': 1.0,'ls': 'solid', 'color':'orange'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'maroon'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'darkblue'},
                        {'lw': 2.0,'ls': 'dashed', 'color':'darkorange'},
                ],
                contour_colors=['lightcoral','dodgerblue','orange','maroon','darkblue','darkorange'],
                contour_ls=['solid','solid','solid','dashed','dashed','dashed'], 
                contour_lws=[1.0,1.0,1.0,2.0,2.0,2.0],
                filled=[True,True,True,False,False,False],
                #params=['logA','omegam','omegab','H0','ns'])#'IA1','IA2'])#'DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
                params=['sigma8','omegam'])

g.export('plots/shifted_chains_sigma8.pdf')








