import numpy as np
from scipy.integrate import RK45
from numpy.linalg import inv
import numpy.linalg as linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize,least_squares, fmin_bfgs
from tabulate import tabulate
from scipy.linalg import expm, sinm, cosm
from sympy import *
import sympy as sp
import pickle
import os
import random

from functions import trajectories, fitfunc, getMat

##########################################################################################################################################
###################################### Defining time axis and loading optimal flip angle sequences ####################################### 
##########################################################################################################################################

TR = 2
endp = 96
tspan = np.squeeze(np.linspace(0,(endp-1)*TR,int((endp-1)*TR/TR+1)).reshape([-1,1]))

VFA_seq = []
# Optimized constant flip angle
VFA_seq.append(np.array(np.repeat(13.4741,len(tspan))))
# Variable flip angle schemes 
with open("./Res_Vk", "rb") as fp:
     Res = pickle.load(fp)
ResT = []
for i in range(0,len(Res)-1):
    if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15: #  and len(Res[i]) != 15
        ResT.append(Res[i])
VFA_seq.append(ResT[Res[25][0]].x)
with open("./Res_VR1S", "rb") as fp:
     Res = pickle.load(fp)
ResT = []
for i in range(0,len(Res)-1):
    if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15:
        ResT.append(Res[i])
VFA_seq.append(ResT[Res[25][0]].x)
with open("./Res_VR1P", "rb") as fp:
     Res = pickle.load(fp)
ResT = []
for i in range(0,len(Res)-1):
    if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15:
        ResT.append(Res[i])
VFA_seq.append(ResT[Res[25][0]].x)
with open("./Res_Vall", "rb") as fp:
     Res = pickle.load(fp)
ResT = []
for i in range(0,len(Res)-1):
    if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15:
        ResT.append(Res[i])
VFA_seq.append(ResT[Res[25][0]].x)
with open("./Res_Vsubopt", "rb") as fp:
     Res = pickle.load(fp)
ResT = []
for i in range(0,len(Res)-1):
    if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15:
        ResT.append(Res[i])
VFA_seq.append(ResT[Res[25][0]].x)

########################################################################################################
############################### Plotting the optimal flip angle sequence ###############################
########################################################################################################

seq_name = ['CFA-$k$','VFA-$k$','VFA-$R_{1S}$','VFA-$R_{1P}$','VFA-all','VFA-suboptimal']
color_name = ['firebrick', 'darkcyan', 'green', 'royalblue','palevioletred','goldenrod']

fig, ax_dict = plt.subplot_mosaic([['p0','p1','p2'], ['p3','p4', 'p5']],empty_sentinel="BLANK",dpi=600, gridspec_kw={
        "wspace": 0.01,
        "hspace": 0.3,
    })
fig.set_size_inches(15, 7.5, forward=True)

for i in range(0,6):
    ax_dict['p'+str(i)].plot(tspan,VFA_seq[i],color=color_name[i])
    ax_dict['p'+str(i)].set_ylim((-5,95))
    ax_dict['p'+str(i)].set_yticks((0,30,60,90))
    ax_dict['p'+str(i)].plot(np.array([0,196]),np.array([90,90]),'--k')
    ax_dict['p'+str(i)].set_xlabel('Time [s]')
    ax_dict['p'+str(i)].set_title(seq_name[i])
    ax_dict['p'+str(i)].grid()
    ax_dict['p'+str(i)].set_yticklabels([])
ax_dict['p0'].set_ylabel('Flip angle [degrees]')
ax_dict['p3'].set_ylabel('Flip angle [degrees]')
ax_dict['p0'].set_yticklabels([0,30,60,90])
ax_dict['p3'].set_yticklabels([0,30,60,90])
plt.show()

#################################################################################################
################################# Monte Carlo - Varying parameters ##############################
#################################################################################################

parameters = np.array([0.0135, 1/35, 1/54, 5, 0.1])
parameters_t = np.array([0.0135, 1/35, 1/54, 5, 0.1])

noise_sd = np.array([5e-03,5e-03])
noise_mu = 0

MC = 1000

loop = [-0.5, -0.25, 0, 0.25, 0.5]
for a in range(0,len(VFA_seq)):
    alpha = VFA_seq[a]
    for p in range(len(parameters)):
        for l in range(0,len(loop)):
            parameters_t = np.array([0.0135, 1/35, 1/54, 5, 0.1, 1])
            parameters_t[p] = parameters[p]+parameters[p]*loop[l]
            alpha_t = alpha * parameters_t[5]
            [Mxy,Mz] = trajectories(alpha_t,parameters_t[0:5],TR)
            
            res = []
            diff = []
            for i in range(0,MC):
                
                np.random.seed(100+i)
                noise = np.random.normal(noise_mu,noise_sd,[len(tspan),2]).T
                data = (Mxy + noise)/(Mxy[0,0])
                
                f = lambda x: fitfunc(alpha, x, TR, data)
                
                param = least_squares(f, parameters[0:5], jac='2-point', method='trf', 
                                  ftol=1e-16, xtol=1e-15, gtol=1e-16, x_scale=1.0, loss='linear', 
                                  f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, 
                                  jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
                if param.success == False: print('Optimization failed...'+str(a)+str(p)+str(l))
                res.append(param)
                diff.append(param.x[0]-parameters_t[0])
            with open("./MonteCarloRes/resEAAP_{}_{}_{}".format(a,p,l), "wb") as fp:
                pickle.dump(res, fp)

#########################################################################################################
############################### Plotting distribution function ##########################################
#########################################################################################################

par_name = ['$k$','$R_{1S}$','$R_{1P}$','$S_{0}$','$P_{0}$','$B_1^S$']
par_space = ['                                 ','                              ','                                ','                          ','                                ']
color_list = ['firebrick','darkcyan','green','royalblue','palevioletred','goldenrod']

parameters = np.append(np.array([0.0135, 1/35, 1/54, 5, 0.1]),1)
parameters_t = np.append(np.array([0.0135, 1/35, 1/54, 5, 0.1]),1)

MC = 1000
paramfit = np.zeros([len(VFA_seq),len(parameters)-1,MC*(len(parameters)-1)])

for pp in range(0,len(parameters)-1):
    for aa in range(len(VFA_seq)):
        with open("./MonteCarloRes/resEAAP_{}_{}_{}".format(aa,pp,2), "rb") as fp:
            res = pickle.load(fp)
            for n in range(0,MC):
                paramfit[aa,:,n+pp*MC]=res[n].x
                if res[n].success == False: print('failed')

#################################################################################################################
############################ Plotting distribution function of rates only #######################################
#################################################################################################################

fig, ax_dict = plt.subplot_mosaic([['p0','p1','p2']],dpi = 600,empty_sentinel="BLANK", gridspec_kw={
        "wspace": 0.15,
        "hspace": 0.3,
    })
fig.set_size_inches(15, 7.5/2, forward=True)
fig.suptitle("Distribution of Parameters", fontsize=18)
for pp in range(0,3):
    for aa in range(0,len(VFA_seq)):
        ax_dict['p'+str(pp)].plot(np.linspace(np.min(paramfit[:,pp,:]), np.max(paramfit[:,pp,:])),
                                  stats.gaussian_kde(paramfit[aa,pp,:], bw_method=None, weights=None).
                                  pdf(np.linspace(np.min(paramfit[:,pp,:]), np.max(paramfit[:,pp,:])))
                                  ,color = color_list[aa],label = seq_name[aa])
        ax_dict['p'+str(pp)].axvline(x=np.array([0.0135, 1/35, 1/54, 0, 0])[pp],color='k')
        x = np.array([np.mean(np.linspace(np.min(paramfit[:,pp,:]), np.max(paramfit[:,pp,:]))[0:2]),
                      np.array([0.0135, 1/35, 1/54, 0, 0])[pp],
                      np.mean(np.linspace(np.min(paramfit[:,pp,:]), np.max(paramfit[:,pp,:]))[-2:])])
        if pp == 0: s = 1e+2;st = '1e-2'
        elif pp == 1: s = 1e+2;st = '1e-2'
        elif pp == 2: s = 1e+2;st = '1e-2'
        ax_dict['p'+str(pp)].set_xticks(x,np.round(x*s,2))
        ax_dict['p'+str(pp)].set_xlabel(str(st), loc='right') # par_name[pp] + par_space[pp] + 
        ax_dict['p'+str(pp)].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_dict['p0'].set_ylabel('Density')
ax_dict['p2'].legend(bbox_to_anchor=(1.05, 0.75),loc='upper left', borderaxespad=0.2, title="Schemes")
ax_dict['p0'].text(0.01347, -470, '$k$',fontsize = 20)
ax_dict['p1'].text(0.0284, -350, '$R_{1S}$',fontsize = 20)
ax_dict['p2'].text(0.0182, -260, '$R_{1P}$',fontsize = 20)
plt.show()

########################################################################################################
########################################### Robustness plot ############################################
########################################################################################################

par_space = ['                                 ','                              ','                             ','                                      ','                                     ']
RMSE = np.zeros([len(VFA_seq),len(loop)])

fig, ax_dict = plt.subplot_mosaic([['p0','p1','p2'], ['p3','p4', 'BLANK']],dpi = 600,empty_sentinel="BLANK", gridspec_kw={
        "wspace": 0.15,
        "hspace": 0.4,
    })
fig.set_size_inches(15, 7.5, forward=True)

for pp in range(0,len(parameters)-1):
    for ll in range(0,len(loop)):
        for aa in range(0,len(VFA_seq)):
            with open("./MonteCarloRes/resEAAP_{}_{}_{}".format(aa,pp,ll), "rb") as fp:
                res = pickle.load(fp)
            parameters_t = np.array([0.0135, 1/35, 1/54, 5, 0.1, 1])
            parameters_t[pp] = parameters[pp]+parameters[pp]*loop[ll]
            kfit = np.zeros(len(res))
            for i in range(len(res)):
                kfit[i] = res[i].x[0]
            RMSE[aa,ll] = np.sqrt(np.mean(np.array(parameters_t[0] - kfit)**2))
            
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[0,])-np.log(RMSE[0,]),label = seq_name[0],color='firebrick',marker = 'o', linewidth=2, markersize=6)
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[1,])-np.log(RMSE[0,]),label = seq_name[1],color='darkcyan',marker = 'o', linewidth=2, markersize=6)
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[2,])-np.log(RMSE[0,]),label = seq_name[2],color='green',marker = 'o', linewidth=2, markersize=6)
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[3,])-np.log(RMSE[0,]),label = seq_name[3],color='royalblue',marker = 'o', linewidth=2, markersize=6)
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[4,])-np.log(RMSE[0,]),label = seq_name[4],color='palevioletred',marker = 'o', linewidth=2, markersize=6)
    ax_dict['p'+str(pp)].plot(parameters[pp]+parameters[pp]*np.array(loop),np.log(RMSE[5,])-np.log(RMSE[0,]),label = seq_name[5],color='goldenrod',marker = 'o', linewidth=2, markersize=6)
    x = np.array(parameters[pp]+parameters[pp]*np.array(loop))
    if pp == 0: s = 1e+2;st = '1e-2'
    elif pp == 1: s = 1e+2;st = '1e-2'
    elif pp == 2: s = 1e+2;st = '1e-2'
    elif pp == 3: s = 1;st = ' '
    elif pp == 4: s = 1;st = ' '
    ax_dict['p'+str(pp)].set_xticks(x,np.round(x*s,2))
    ax_dict['p'+str(pp)].set_xlabel(str(st), loc='right') # par_name[pp] + par_space[pp] + 
ax_dict['p0'].set_ylabel('CFA-$k$ offset $k$ $log(RMSE)$')
ax_dict['p3'].set_ylabel('CFA-$k$ offset $k$ $log(RMSE)$')
ax_dict['p0'].text(0.0135-0.03*0.0135, -0.53, '$k$',fontsize = 20)
ax_dict['p1'].text(1/35-0.03*(1/35), -0.53, '$R_{1S}$',fontsize = 20)
ax_dict['p2'].text(1/54-0.03*(1/54), -0.55, '$R_{1P}$',fontsize = 20)
ax_dict['p3'].text(5-0.03*5, -0.53, '$S_0$',fontsize = 20)
ax_dict['p4'].text(0.1-0.03*0.1, -0.53, '$P_0$',fontsize = 20)
ax_dict['p4'].legend(bbox_to_anchor=(1.5, 0.75),loc='upper left', borderaxespad=0.2, title="Schemes")
plt.show()

#################################################################################################
######################### Monte Carlo simulations B1S fixed #####################################
#################################################################################################
from functions_B import trajectories, fitfunc

parameters = np.array([0.0135, 1/35, 1/54, 5, 0.1, 1])

noise_sd = 5e-03
noise_mu = 0
MC = 1000

for a in range(0,len(VFA_seq)):
    alpha = VFA_seq[a]
    [Mxy,Mz] = trajectories(alpha,parameters[0:6],TR)
    
    res0 = []
    res1 = []
    for i in range(0,MC):
        
        np.random.seed(200+i)
        noise = np.random.normal(noise_mu,noise_sd,[len(tspan),2]).T
        data = (Mxy + noise)/(Mxy[0,0])
        
        l = 0
        from functions import trajectories, fitfunc
        f = lambda x: fitfunc(alpha, x, TR, data)
        param0 = least_squares(f, parameters[0:5], jac='2-point', method='trf', 
                      ftol=1e-16, xtol=1e-15, gtol=1e-16, x_scale=1.0, loss='linear', 
                      f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, 
                      jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
        if param0.success == False: print('Optimization failed...'+str(a)+str(p)+str(l))
        
        res0.append(param0)
        
        l = 1
        from functions_B import trajectories, fitfunc
        f = lambda x: fitfunc(alpha, x, TR, data)
        param1 = least_squares(f, parameters[0:6], jac='2-point', method='trf', 
                      ftol=1e-16, xtol=1e-15, gtol=1e-16, x_scale=1.0, loss='linear', 
                      f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, 
                      jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
        if param1.success == False: print('Optimization failed...'+str(a)+str(p)+str(l))
        
        res1.append(param1)
        
    # sequence number, parameter, parameter loop
    with open("./MonteCarloRes/resFixB1_{}_{}".format(a,0), "wb") as fp:
        pickle.dump(res0, fp)
    with open("./MonteCarloRes/resFixB1_{}_{}".format(a,1), "wb") as fp:
        pickle.dump(res1, fp)

#################################################################################################
######################### Monte Carlo simulations B1S varying ###################################
#################################################################################################

from functions_B import trajectories, fitfunc

noise_sd = 5e-03
noise_mu = 0
MC = 1000

for a in range(0,len(VFA_seq)):
    alpha = VFA_seq[a]
    
    res0 = []
    res1 = []
    for i in range(0,MC):
        np.random.seed(i)
        B1S = np.random.normal(1,0.15,size=1)
        [Mxy,Mz] = trajectories(alpha,np.append(parameters[0:5], parameters[5]*B1S),TR)
        np.random.seed(300+i)
        noise = np.random.normal(noise_mu,noise_sd,[len(tspan),2]).T
        data = (Mxy + noise)/(Mxy[0,0])
        
        l = 0
        from functions import trajectories, fitfunc
        f = lambda x: fitfunc(alpha, x, TR, data)
        param0 = least_squares(f, parameters[0:5], jac='2-point', method='trf', 
                      ftol=1e-16, xtol=1e-15, gtol=1e-16, x_scale=1.0, loss='linear', 
                      f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, 
                      jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
        if param0.success == False: print('Optimization failed...'+str(a)+str(p)+str(l))
        
        param0.B1S = B1S
        param0.noise = noise
        
        res0.append(param0)
        
        l = 1
        from functions_B import trajectories, fitfunc
        f = lambda x: fitfunc(alpha, x, TR, data)
        param1 = least_squares(f, parameters[0:6], jac='2-point', method='trf', 
                      ftol=1e-16, xtol=1e-15, gtol=1e-16, x_scale=1.0, loss='linear', 
                      f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, 
                      jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
        if param1.success == False: print('Optimization failed...'+str(a)+str(p)+str(l))
        param1.B1S = B1S
        param1.noise = noise
        
        res1.append(param1)
        
    # sequence number, parameter, parameter loop
    with open("./MonteCarloRes/resVarB1_{}_{}".format(a,0), "wb") as fp:
        pickle.dump(res0, fp)
    with open("./MonteCarloRes/resVarB1_{}_{}".format(a,1), "wb") as fp:
        pickle.dump(res1, fp)

#################################################################################################
###################### Mapping simulation results into matrices #################################
#################################################################################################

MC = 1000

paramfitF = np.zeros([len(VFA_seq),2,len(parameters),MC])

for l in range(0,2):
    for a in range(0,len(VFA_seq)):
        with open("./MonteCarloRes/resFixB1_{}_{}".format(a,l), "rb") as fp: 
            res = pickle.load(fp)
            for n in range(0,len(res)):
                if res[n].success == True:
                    paramfitF[a,l,:len(res[0].x),n] = res[n].x
                elif res[n].success == False:
                    paramfitF[a,l,:len(res[0].x),n] = nan
            ind = np.arange(0, len(res))[~np.logical_and(np.mean(paramfitF[a,l,0,:])-np.nanstd(paramfitF[a,l,0,:])*5 < paramfitF[a,l,0,:],paramfitF[a,l,0,:] < np.mean(paramfitF[a,l,0,:])+np.nanstd(paramfitF[a,l,0,:])*5)]
            paramfitF[a,l,:,ind] = nan

paramfitF = np.delete(paramfitF, 3,2)
paramfitF = np.delete(paramfitF, 3,2)

paramfitV = np.zeros([len(VFA_seq),2,len(parameters)+1,MC])
for l in range(0,2):
    for a in range(0,len(VFA_seq)):
        with open("./MonteCarloRes/resVarB1_{}_{}".format(a,l), "rb") as fp:
            res = pickle.load(fp)
            for n in range(0,len(res)):
                if res[n].success == True:
                    paramfitV[a,l,:len(res[0].x),n] = res[n].x
                    paramfitV[a,l,-1,n] = np.float64(res[n].B1S)
                elif res[n].success == False:
                    paramfitV[a,l,:len(res[0].x),n] = nan
            ind = np.arange(0, len(res))[~np.logical_and(np.mean(paramfitV[a,l,0,:])-np.nanstd(paramfitV[a,l,0,:])*5 < paramfitV[a,l,0,:],paramfitV[a,l,0,:] < np.mean(paramfitV[a,l,0,:])+np.nanstd(paramfitV[a,l,0,:])*5)]
            paramfitV[a,l,:,ind] = nan

paramfitV = np.delete(paramfitV, 3,2)
paramfitV = np.delete(paramfitV, 3,2)

#################################################################################################
############################## Standard deviation comparison plot ###############################
#################################################################################################

par_name = ['$k$      ','$R_{1S}$      ','$R_{1P}$      ','$B_1^S$      ']
label_text = ['Model without $B_1^S$MC with fixed $B_1^S$', 
              'Model with $B_1^S$ MC with fixed $B_1^S$', 
              'Model without $B_1^S$ MC with varying $B_1^S$', 
              'Model with $B_1^S$ MC with varying $B_1^S$']
seq_name = ['CFA-$k$','VFA-$k$','VFA-$R_{1S}$','VFA-$R_{1P}$','VFA-all','VFA-suboptimal']

fig, axs = plt.subplot_mosaic([[0],[1],[2]],empty_sentinel="BLANK", gridspec_kw={
        "wspace": 0.15,
        "hspace": 0.0,
    },dpi=150)

fig.set_size_inches(5.5, 7.5, forward=True)
x_dev = np.array([0,0.2,0.2,0.2,0.2,0.2])
x = np.array([0,1,2,3,4,5])
dev_fac = np.array([1.5,2.5,1.5])
exp = 1
for pm in range(3):
    axs[pm].errorbar(x-x_dev, np.repeat(parameters[pm],6), yerr=np.nanstd(paramfitF[:,0,pm,:],1)**exp, capsize=4, label=label_text[0], fmt='', color = 'blue')
    axs[pm].errorbar(x-0.0, np.repeat(parameters[pm],6), yerr=np.concatenate((np.array([0]),np.nanstd(paramfitF[1:,1,pm,:],1)**exp)), capsize=4, label=label_text[1], fmt='', color = 'orange')
    axs[pm].errorbar(x+0.2, np.repeat(parameters[pm],6), yerr=np.concatenate((np.array([0]),np.nanstd(paramfitV[1:,1,pm,:],1)**exp)), capsize=4, label=label_text[3], fmt='', color = 'green')
    axs[pm].ticklabel_format(useOffset=False)
    axs[pm].errorbar(np.array([-0.25,5.25]), np.array([parameters[pm],parameters[pm]]), color ='k')
    if pm < 3: axs[pm].set_ylim((parameters[pm]-np.nanstd(paramfitF[:,0,pm,:],1)[0]**exp*dev_fac[pm],parameters[pm]+np.nanstd(paramfitF[:,0,pm,:],1)[0]**exp*dev_fac[pm]))
    axs[pm].set_ylabel(par_name[pm],fontsize=14, rotation='horizontal')
axs[1].legend(loc='center right',bbox_to_anchor=(1.7, 0.5))
axs[2].set_xlabel('Scheme',fontsize=14)
axs[0].set_title('Standard deviations of parameter estimates',fontsize=16)
axs[2].set_xticks(np.array([0,1,2,3,4,5]),('CFA-$k$','VFA-$k$','VFA-$R_{1S}$','VFA-$R_{1P}$','VFA-all','VFA-sub-\noptimal'))
plt.show()

fig, axs = plt.subplot_mosaic([[0],[1],[2]],empty_sentinel="BLANK", gridspec_kw={
        "wspace": 0.15,
        "hspace": 0.0,
    },dpi=150)

fig.set_size_inches(6, 7.5, forward=True)
x = np.array([0,1,2,3,4,5])
dev_fac = np.array([2,17,8])
for pm in range(3):
    axs[pm].errorbar(x-0.2, np.repeat(parameters[pm],6), yerr=np.nanstd(paramfitF[:,0,pm,:],1)**exp, capsize=5, label=label_text[0], fmt='', color = 'blue')
    axs[pm].errorbar(x-0.075, np.repeat(parameters[pm],6), yerr=np.concatenate((np.array([0]),np.nanstd(paramfitF[1:,1,pm,:],1)**exp)), capsize=5, label=label_text[1], fmt='', color = 'orange')
    axs[pm].errorbar(x+0.075, np.repeat(parameters[pm],6), yerr=np.nanstd(paramfitV[:,0,pm,:],1)**exp, capsize=5, label=label_text[2], fmt='', color = 'red')
    axs[pm].errorbar(x+0.2, np.repeat(parameters[pm],6), yerr=np.concatenate((np.array([0]),np.nanstd(paramfitV[1:,1,pm,:],1)**exp)), capsize=5, label=label_text[3], fmt='', color = 'green')
    axs[pm].ticklabel_format(useOffset=False)
    axs[pm].errorbar(np.array([-0.25,5.25]), np.array([parameters[pm],parameters[pm]]), color ='k')
    if pm < 3: axs[pm].set_ylim((parameters[pm]-np.nanstd(paramfitF[:,0,pm,:],1)[0]*dev_fac[pm],parameters[pm]+np.nanstd(paramfitF[:,0,pm,:],1)[0]*dev_fac[pm]))
    axs[pm].set_ylabel(par_name[pm],fontsize=14, rotation='horizontal')
axs[1].legend(loc='center right',bbox_to_anchor=(1.7, 0.5))
axs[2].set_xlabel('Scheme',fontsize=14)
axs[0].set_title('Standard deviations of parameter estimates',fontsize=16)
axs[2].set_xticks(np.array([0,1,2,3,4,5]),('CFA-$k$','VFA-$k$','VFA-$R_{1S}$','VFA-$R_{1P}$','VFA-all','VFA-sub-\noptimal'))
plt.show()
#################################################################################################
###################### Mapping simulation results into matrices #################################
#################################################################################################

paramfitV = np.zeros([len(VFA_seq),2,len(parameters),MC])
for a in range(0,len(VFA_seq)):
    for l in range(0,2):
        with open("./MonteCarloRes/resVarB1_{}_{}".format(a,l), "rb") as fp:
            res = pickle.load(fp)
            for n in range(0,MC):
                if res[n].success == True and ~np.any(res[n].x[0:3]<0): # 
                    paramfitV[a,l,:len(res[0].x),n] = res[n].x
                    if l==0: paramfitV[a,l,-1,n] = res[n].B1S
                elif res[n].success == False:
                    paramfitV[a,l,:len(res[0].x),n] = nan

par_name = ['$k$','$R_{1S}$','$R_{1P}$','$S_{0}$','$P_{0}$','$B_1^S$']

min_vec = []
max_vec = []

l=0
for a in np.array([0,1,2,3,4,5]):
    ind = np.where(np.logical_and(np.mean(paramfitV[a,l,0,:])-np.nanstd(paramfitV[a,l,0,:])*5 < paramfitV[a,l,0,:],paramfitV[a,l,0,:] < np.mean(paramfitV[a,l,0,:])+np.nanstd(paramfitV[a,l,0,:])*5))
    min_vec.append(np.min(paramfitV[a,l,5,ind]))
    max_vec.append(np.max(paramfitV[a,l,5,ind]))

#################################################################################################
#################################### Plotting scatterplots  #####################################
#################################################################################################

fig = plt.figure(constrained_layout=True, figsize=(33/2,22/2),dpi=600)
subfigs = fig.subfigures(2, 3)
for a, subfig in enumerate(subfigs.flat):
    ax = subfig.subplots(3, 3)
    ind = np.where(np.logical_and(np.mean(paramfitV[a,l,0,:])-np.nanstd(paramfitV[a,l,0,:])*5 < paramfitV[a,l,0,:],paramfitV[a,l,0,:] < np.mean(paramfitV[a,l,0,:])+np.nanstd(paramfitV[a,l,0,:])*5))
    for n in np.array([0,1,2]):
        for m in np.array([0,1,2]):
            ax[m,n].scatter(paramfitV[a,l,n,ind],paramfitV[a,l,m,ind],c=paramfitV[a,l,5,ind], vmin=np.min(min_vec), vmax=np.max(max_vec))
            ax[m,0].set_ylabel(par_name[m],fontsize=15)
            if n!=0: ax[m,n].set_yticks(())
            if m!=2: ax[m,n].set_xticks(())
        ax[2,n].set_xlabel(par_name[n],fontsize=15)
    subfig.suptitle(seq_name[a], fontsize=20)
cax = fig.add_axes([1.02,0.1,0.03,0.8])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(min_vec), np.max(max_vec)), cmap=mpl.cm.viridis),
              orientation='vertical',cax=cax)
cbar.set_label(label='$B_1^S$',size=20)
plt.show()


