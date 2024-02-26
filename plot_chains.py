import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt
import os

# debug file to plot samples used for training in a triangle plot

ranges={'logAs':(1.61,3.91),
        'ns':(0.87,1.07),
        'H0':(55,91),
        'Omegab':(0.03,0.07),
        'Omegam':(0.1,0.9)
}

def emu_to_params(theta,means=None):
    logAs = theta[:,0]
    ns    = theta[:,1]
    H0    = theta[:,2]
    ombh2 = theta[:,3]
    omch2 = theta[:,4]
    dz1   = theta[:,5]
    dz2   = theta[:,6]
    dz3   = theta[:,7]
    dz4   = theta[:,8]
    dz5   = theta[:,9]
    IA1   = theta[:,10]
    IA2   = theta[:,11]
    try:
        M1    = theta[:,12]
        M2    = theta[:,13]
        M3    = theta[:,14]
        M4    = theta[:,15]
        M5    = theta[:,16]
    except:
        M1 = np.zeros(len(theta))
        M2 = np.zeros(len(theta))
        M3 = np.zeros(len(theta))
        M4 = np.zeros(len(theta))
        M5 = np.zeros(len(theta))

    
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    
    h = H0/100
    
    omb = ombh2/(h**2)
    omc = omch2/(h**2)
    omn = omnh2/(h**2)
    
    omm = omb+omc+omn
    ommh2 = omm*(h**2)
        
    return np.transpose(np.array([logAs,ns,H0,omb,omm,dz1,dz2,dz3,dz4,dz5,IA1,IA2,M1,M2,M3,M4,M5]))

#base_path = './projects/lsst_y1/emulator_output/chains/'
names = ['logAs','ns','H0','Omegab','Omegam']#,'dz1','dz2','dz3','dz4','dz5','IA1','IA2']

training_chain = np.load('./chains/params_with_sigma8_training.npy')
print(training_chain[0])
samples = emu_to_params(training_chain)
mcmc_chain1 = getdist.mcsamples.MCSamples(samples=samples[:,:5],names=names,ranges=ranges)
#mcmc_chain1.removeBurnFraction(0.5)
#mcmc_chain1.thin(10)

base_path = '/home/grads/extra_data/evan/shifted_chains/'
chains = os.listdir(base_path)[::50] # do we really need all 700+ chains?
chain_list=[mcmc_chain1]
colors = ['#ff0000ff']

for i in range(20):
    chain_list.append(mcmc_chain1)
    colors.append('#ff0000ff')

# names = ['sigma8','ns','H0','Omegab','Omegam']#,'dz1','dz2','dz3','dz4','dz5','IA1','IA2']
i=0
for file in chains:
    if '.txt' not in file:
        continue
    mcmc_chain = getdist.mcsamples.loadMCSamples(base_path+file[:-4],no_cache=True)
    print(mcmc_chain.samples[0])
    samples = emu_to_params(mcmc_chain.samples)
    mcmc_chain.setSamples(samples)
    # print(mcmc_chain.samples[0])
    # print(mcmc_chain.getParamNames().getRunningNames())
    # mcmc_chain.removeBurnFraction(0.5)
    # mcmc_chain.thin(10)
    # p = mcmc_chain.getParams()
    # mcmc_chain.addDerived((p.omegab+p.omegac+(3.046/3)**(3/4)*0.06/94.1)*(100**2)/(p.H0**2),name='Omegam',label='\Omega_m')
    # p = mcmc_chain.getParams()
    # mcmc_chain.addDerived(((np.exp(p.logAs)/(10**10))/3.135e-9)**(1/2) * \
    #               (p.omegab/0.024)**(-0.272) * \
    #               (p.Omegam*(p.H0)**2/((100**2)*0.14))**(0.513) * \
    #               (3.123*p.H0/100)**((p.ns-1)/2) * \
    #               (p.H0/72)**(0.698) * \
    #               (p.Omegam/0.27)**(0.236) * \
    #               (1-0.014),name='sigma8',label='\sigma_8')
    chain_list.append(mcmc_chain)
    print(len(mcmc_chain.samples))
    colors.append('#0000000d')
    print(i)
    i+=1

print('plotting...')
plot_names=names = ['logAs','ns','H0','Omegab','Omegam']#
g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.settings.line_labels=False
g.triangle_plot(chain_list,
               filled=False,
               contour_colors=colors,
               contour_ls='solid', 
               contour_lws=1,
               contour_args={'alpha': 0.05},
               params=plot_names,
               ranges=ranges)
g.export('plots/noise_chains.pdf')