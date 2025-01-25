import numpy as np
from numpy.linalg import inv
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize,least_squares, fmin_bfgs
import scipy.stats as stats
from scipy.linalg import expm, sinm, cosm
from sympy import *
from sympy import exp

def trajectories(alpha,parameters,TR):
    """
    trajectories generates magnetization trajectories
    
    :alpha: numpy array of flip angle scheme in degrees
    :parameters: numpy array of model parameters in the order k, R_1S, R_1P, S0, P0 and B_1^S
    :TR: Repitition time
    :return: x longitudinal magnetization trajectory of the substrate and product, y transversial magnetization trajectory of the substrate and product
    """ 
    N = len(alpha)
    alpha = (np.pi*alpha/180)*parameters[len(parameters)-1]
    
    A = np.array([[-parameters[0]-parameters[1],0], [parameters[0], -parameters[2]]])
    G = expm(A*TR)
    
    n = np.size(G,0)
    m = np.size(G,0)
    x = np.zeros([n, N]) 
    y = np.zeros([m, N])
    
    x[0, 0] = parameters[3] # inserting the starting value of substrate
    x[1, 0] = parameters[4] # inserting the starting value of product
    
    for t in range(0,N-1):
        x[:,t+1] = np.squeeze(np.array(G * np.cos(alpha[t]) @ np.array([x[:,t]]).T),1)
    y = (x.T * np.sin(alpha)[:, np.newaxis]).T
        
    return [y,x]

def fitfunc(alpha, parameters, TR, data):
    """
    Computes residuals between data and generated trajectories
    
    :alpha: Numpy array of flip angle scheme in degrees
    :parameters: numpy array of model parameters in the order k, R_1S, R_1P, S0 and P0
    :TR: Repitition time
    :data: Numpy array containing data trajectories of compounds 
    :return: Vector of residuals consisting of differences between data and trajectories generated based on alpha, parameters and TR
    """ 
    [Mxy,Mz] = trajectories(alpha,parameters,TR)
    err =  np.ravel((Mxy - data),order = 'C')
    
    return err



