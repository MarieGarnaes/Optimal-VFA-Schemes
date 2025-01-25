import numpy as np
from numpy.linalg import inv
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from scipy.optimize import minimize,least_squares, fmin_bfgs
from scipy.linalg import expm, sinm, cosm
from sympy import *
import sympy as sy
import time
import pickle
import multiprocessing
import random
from functions import trajectories, FIM, fitfunc, getMat

##################################################################################################
##################################### Defining parameters  ####################################### 
##################################################################################################

# Non-dimensionalization
T0 = 2
M0 = 1
TR = 2 / T0
N = 96
endp = int(96)
tspan = np.squeeze(np.linspace(0,(endp-1)*TR,int((endp-1)*TR/TR+1)).reshape([-1,1]))
std = np.array([5e-3,5e-3])
# Upper and lower bounds for flip angles
lb = 0
ub = 90
bnds = ((0.1 , ub),)+(((lb, ub),) * (len(tspan)-1))

# Defining parameters and vectors for implementation of prior
b_sd = 0.15
b_mu = 1
b = np.linspace(0.5, 1.5, num=11)
q = 1/(np.sqrt(2*np.pi*b_sd**2))*np.exp(-(b-b_mu)**2/(2*b_sd**2))

# Function that plots the progress of the optimization
def plotf(x):
    global fv
    global bestfv
    global ii
    global t0
    print(f'f: {fv}, fvals/sec:{ii/(time.time()-t0)}')

#######################################################################################################
############################### Optimizing constant flip angle sequence (CFA) #########################
#######################################################################################################

parameters = np.array([0.0135, 1/35, 1/54, 5, 0.1])*np.array([T0,T0,T0,M0,M0])
W = np.eye(len(parameters))*np.array([1/(parameters[0]**2),0,0,0,0])
[G,Gdd] = getMat(parameters,TR)

def f(x):
    return np.trace(np.linalg.inv(FIM(np.repeat(x,N).T, parameters, TR, std, G, Gdd))/(std[0]**2)*W) 

def O2(x,pool):
    global fv
    global bestfv
    global ii
    ii+=1
    xx=[x*bi for bi in b]
    ff=pool.map(f,xx)
    
    fv=np.sum([q[i]*ff[i] for i in range(len(q))])
    if fv<bestfv:
        global bestx
        bestx=x
        bestfv=fv
    return fv

ResC = []
for i in np.linspace(10,80,8):
    if __name__ == '__main__':
        global fv
        global bestfv
        global bestx
        bestfv=np.inf
        global ii
        ii=0                
        pool = multiprocessing.Pool()
        global t0
        t0=time.time()
        resB = scipy.optimize.minimize(O2, x0 = np.array([i]), method='L-BFGS-B', tol = 1e-09*0.04, 
                                        options={'maxiter':10000,'eps':0.0001,'maxls':100,'maxfun':100000},
                                        bounds=((0.1, ub),),
                                        callback=plotf,
                                        args=(pool,)) # tol = 1e-2
        resB.init = i
        resB.G = G
        resB.parameters = parameters
    ResC.append(resB)
    with open("./Res_Ck", "wb") as fp: 
        pickle.dump(ResC, fp)

###############################################################################################
##################### Optimization of variable flip angle sequence ############################
###############################################################################################

for m in range(4,6):
    
    # Defining weights and parameters for specific sequences
    if m == 1: # VFA-k
        w = np.array([1/(parameters[0]**2),0,0,0,0])
        n  = 1
    elif m == 2: # VFA-R1S
        w = np.array([0,1/(parameters[1]**2),0,0,0])
        n  = 1
    elif m == 3: # VFA-R1P
        w = np.array([0,0,1/(parameters[2]**2),0,0])
        n  = 1
    elif m == 4: # VFA-all
        w = np.array([1/(parameters[0]**2),1/(parameters[1]**2),1/(parameters[2]**2),0,0])
        n  = 3
    elif m == 5: # VFA-suboptimal
        parameters = np.array([1.5*0.0135, 1.5*(1/35), 1.5*(1/54), 5, 0.1])*np.array([T0,T0,T0,M0,M0])
        w = np.array([1/(parameters[0]**2),0,0,0,0])
        n  = 1
        [G,Gdd] = getMat(parameters,TR)
    
    tol_level = np.sum(np.multiply(w,parameters))/n # Tolerance level used in minimization function
    W = np.eye(len(parameters))*w
    
    Res = []
    if m > 0:
        
        def f(x):
            return  np.trace(np.linalg.inv(FIM(x.T, parameters, TR, std, G, Gdd))*W)
        
        for i in range(0,25):
            
            def O2(x,pool):
                global fv
                global bestfv
                global ii
                ii+=1
                xx=[np.clip(x,0,90)*bi for bi in b]
                ff=pool.map(f,xx)
                # Defining the objective function including prior and regularization
                fv=np.sum(np.append([(1/n)*q[i]*ff[i]*(1/(std[0]**2)) for i in range(len(q))],0.1*np.sum(np.sin(np.clip(x,0,90)*np.pi/180)**2)))
                
                if fv<bestfv:
                    global bestx
                    bestx=x
                    bestfv=fv
                return fv
            
            st = time.time()
            if __name__ == '__main__':
                global fv
                global bestfv
                global bestx
                bestfv=np.inf
                global ii
                ii=0
                np.random.seed(m*100+i)
                init = np.random.uniform(low=10, high=80, size=len(tspan))
                pool = multiprocessing.Pool()
                global t0
                t0=time.time()
                try:
                    resB = scipy.optimize.minimize(O2, x0 = init, method='L-BFGS-B', tol = 1e-09*tol_level/2, 
                                                   options={'maxiter':10000,'eps':0.0001,'maxls':100,'maxfun':100000},
                                                   bounds=bnds,
                                                   callback=plotf,
                                                   args=(pool,))
                except:
                    resB.mgs = 'exception took place'
            et = time.time()
            print('Number: '+ str(i) + ' Time: '+ str(et-st))
            if resB.success == False: print('Optimization failed...' + resB.message)
            
            resB.init = init
            resB.G = G
            resB.w = w
            resB.parameters = parameters
            Res.append(resB.copy())
            
        # Defining an objective function without regularization
        def f(x):
            return  np.trace(np.linalg.inv(FIM(x.T, parameters, TR, std, G, Gdd))*W)
        O = lambda x: np.sum([q[i]*f(b[i]*np.clip(x,0,90)) for i in range(len(q))])
        
        fun = []; ResT = []
        for i in range(0,len(Res)):
            if Res[i].success == True and Res[i].fun > 0 and len(Res[i]) != 15: # excluding outliers
                ResT.append(Res[i])
                fun.append(O(Res[i].x))
        srt = np.argsort(fun)
        # Determining 
        index_of_small = list(np.array(fun)[srt]).index(min(filter(lambda x : x > 0, list(np.array(fun)[srt]))))
        Res.append([srt[index_of_small]]) 
        if m == 1:
            with open("./Res_Vk", "wb") as fp: 
                pickle.dump(Res, fp)
        elif m == 2:
            with open("./Res_VR1S", "wb") as fp: 
                pickle.dump(Res, fp)
        elif m == 3:
            with open("./Res_VR1P", "wb") as fp: 
                pickle.dump(Res, fp)
        elif m == 4:
            with open("./Res_Vall", "wb") as fp: 
                pickle.dump(Res, fp)
        elif m == 5:
            with open("./Res_Vsubopt", "wb") as fp: 
                pickle.dump(Res, fp)
        



