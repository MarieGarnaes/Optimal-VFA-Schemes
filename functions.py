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
    :parameters: numpy array of model parameters in the order k, R_1S, R_1P, S0 and P0
    :TR: Repitition time
    :return: x longitudinal magnetization trajectory of the substrate and product, y transversial magnetization trajectory of the substrate and product
    """ 
    
    alpha = (np.pi*alpha/180) # degrees to radians conversion
    A = np.array([[-parameters[0]-parameters[1],0], [parameters[0], -parameters[2]]])
    G = expm(A*TR)
    
    n = np.size(G,0)
    m = np.size(G,0)
    x = np.zeros([n, len(alpha)]) 
    y = np.zeros([m, len(alpha)])
    
    x[0, 0] = parameters[3] # inserting the starting value of substrate
    x[1, 0] = parameters[4] # inserting the starting value of product
    
    for t in range(0,len(alpha)-1):
        x[:,t+1] = np.squeeze(np.array(G * np.cos(alpha[t]) @ np.array([x[:,t]]).T),1)
    y = (x.T * np.sin(alpha)[:, np.newaxis]).T
    
    return [y,x]

def FIM(alpha,parameters,TR,std,G,Gdd):
    """
    Fisher Information Matrix (FIM) computation
    
    :alpha: numpy array of flip angle scheme in degrees
    :parameters: numpy array of model parameters in the order k, R_1S, R_1P, S0 and P0
    :TR: Repitition time
    :std: standard deviation of noise for each of the respective compounds
    :G: array containing transistion matrix (computed in the function getMat)
    :Gdd: array containing G-matrix elements differentiated with respect to each of the parameters (computed in the function getMat)
    :return: The Fisher Information Matrix
    """ 
    [y,x] = trajectories(alpha,parameters,TR)
    alpha = np.pi*alpha/180
    # Normalization factor N
    N = 1#/(parameters[3]*np.sin(alpha[0]))
    x = x*N
    
    Dxdp = np.zeros([len(alpha),len(parameters),2])
    Dxdp[0,3,0] = 1
    Dxdp[0,4,1] = 1
    Dydp = np.zeros([len(alpha),len(parameters),2])
    for i in range(0,5): # Parameters
        for n in range(0,len(alpha)-1): # Timesteps
            Dxdp[n+1,i,:] = np.squeeze(Gdd[:,:,i] @ np.array([x[:,n]]).T+G@np.array([Dxdp[n,i,:]]).T,1)*np.cos(alpha[n]) 
        Dydp[:,i,:] = (Dxdp[:,i,:]*np.sin(alpha)[:, np.newaxis])
    
    var_inv = np.array([1/((std[0]*N)**2),1/((std[1]*N)**2)])
    FIM = np.zeros([5,5])
    for i in range(0,5):
        for j in range(0,5):
            FIM[i,j] = np.sum(np.multiply(Dydp[:,i,:], Dydp[:,j,:]) * var_inv)
    return FIM

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


def getMat(parameters,TR):
    """
    Generates matrices needed for FIM computation
    
    :parameters: numpy array of model parameters in the order k, R_1S, R_1P, S0 and P0
    :TR: Repitition time
    :return: Transistion matrix G, and a matrix Gdd containing the derivatives of G with respect to all parameters individually.
    """ 
    A = np.array([[-parameters[0]-parameters[1],0], [parameters[0], -parameters[2]]])
    G = expm(A*TR)
    # Gd contains differentiated transitionmatrices G
    Gd = np.array([
    [-TR*exp(-TR*(parameters[1] + parameters[0])), 0, -parameters[0]*TR*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) + parameters[0]*TR*exp(TR*(parameters[1] + parameters[0]))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) - parameters[0]*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0])**2 + (-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]),                    0],
    [-TR*exp(-TR*(parameters[1] + parameters[0])), 0,                                                                                               -parameters[0]*TR*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) + parameters[0]*TR*exp(TR*(parameters[1] + parameters[0]))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) - parameters[0]*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0])**2,                    0],
    [                            0, 0,                                                                                                        -parameters[0]*TR*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) - parameters[0]*TR*exp(parameters[2]*TR)*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0]) + parameters[0]*(-exp(parameters[2]*TR) + exp(TR*(parameters[1] + parameters[0])))*exp(-TR*(parameters[2] + parameters[1] + parameters[0]))/(-parameters[2] + parameters[1] + parameters[0])**2, -TR*exp(-parameters[2]*TR)],
    [                            0, 0,                                                                                                                                                                                                                                                                                                                                                                                               0,                    0],
    [                            0, 0,                                                                                                                                                                                                                                                                                                                                                                                               0,                    0]])
    Gdd = np.zeros([len(G),len(G),len(parameters)])
    for g in range(0,len(parameters)):
        Gdd[:,:,g] = np.array([[Gd[g,0],Gd[g,1]],[Gd[g,2],Gd[g,3]]])
    
    return [G,Gdd]