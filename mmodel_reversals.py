'''
Created on 29Jun.,2017

@author: uqdrange
'''
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy import special,stats,optimize
#===============================================================================
# Mixture model Bays (www.paulbays.com)
#===============================================================================
# auxilliary functions
def wrap(theta, bound = np.pi):
    """
    compute wrapped values around the bound
    theta: a scalar or a sequence
    bound: the value around which to wrap, defaults to pi
    """
    theta = np.array(theta)
    wrappedT = (theta+bound)%(bound*2) - bound
    return wrappedT

def cmean(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi):
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        thetaHat = np.arctan2(np.sin(theta).sum(),np.cos(theta).sum())
        return thetaHat

def cstd(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi):
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        R = np.sqrt(np.sin(theta).sum()**2+np.cos(theta).sum()**2)/theta.size
        thetaSD = np.sqrt(-2*np.log(R))
        return thetaSD

def k2sd(K):
    if K == 0:
        S = np.Inf
    elif np.isinf(K):
        S = 0
    else:
        S = np.sqrt(-2*np.log(special.iv(1,K)/special.iv(0,K)))
    return S

def sd2k(S):
    R = np.exp(-S**2/2.)
    if R < .85:
        K = -.4 + 1.39*R + .43/(1 - R)
    elif R < .53:
        K = 2*R + R**3 + (5*R**5)/6.
    else:
        K = 1./(R**3 - 4*R**2 + 3*R)
    return(K)

def A1inv(R):
    if 0 <= R < 0.53:
        K = 2 * R + R**3 + (5 * R**5)/6;
    elif R < 0.85:
        K = -0.4 + 1.39 * R + 0.43/(1 - R);
    else:
        K = 1./(R**3 - 4 * R**2 + 3 * R)
    return K

def llhood(B, X, T, NT):
    '''
    Compute maximum likelihood for:
    1. the given set of parameters (B),
    2. the vector of responses (X),
    3. the vector of target values (T)
    4. the vector of non-target values (NT)

    '''
    B = np.array(B) # N_parameters X 1 array
    X = np.array(X)
    T = np.array(T)
    NT = np.array(NT)

    # reformatting arrays so that they would be 2-dimensional
    if len(X.shape) == 1:
        X = X.reshape(-1,1) # N_trials X 1
    if len(T.shape) == 1:
        T = T.reshape(-1,1) # N_trials X 1
    if len(NT.shape) == 1:
        NT = NT.reshape(-1,1) # N_trials X N_non-targets

    # checking data format
    if ((X.shape[1] > 1 or T.shape[1] > 1 or X.shape[0] != T.shape[0])
        or (~np.isnan(NT).any()
            and (NT.shape[0] != X.shape[0] or NT.shape[0] != T.shape[0]))):
        print('Input is not correctly dimensioned')
        LL = np.nan

    else:
        # formatting non-target array
        n = X.shape[0]
        if np.isnan(NT).any():
            NT = np.zeros([n,1])
            nn = 0
        else:
            nn = NT.shape[1]

        K, M, Pt, Pn = B

        Pu = 1 - Pt - Pn

        if Pu >= 0:
            E = X - T
            # wrapping the difference between the response and the target value
            E = ((E + np.pi)%(2*np.pi)) - np.pi
            NE = X - NT
            # wrapping the difference between the response and the non-target values
            NE = ((NE + np.pi)%(2*np.pi)) - np.pi

            Wt = Pt * stats.vonmises.pdf(E, K, M)
            Wu = Pu * np.ones([n, 1]) / (2 * np.pi)
            Wn = np.zeros(NE.shape)
            if nn > 0:
                Wn = (Pn/nn)*stats.vonmises.pdf(NE, K, M)

            W = np.hstack([Wt,Wn,Wu]).sum(1).reshape(-1,1)

            LL = np.log(W).sum()
        else:
            LL = -1e10

    return -LL

def mmfit(X = np.nan,T = np.nan, NT = np.nan, iter = 100):

    X = np.array(X)
    T = np.array(T)
    NT = np.array(NT)

    # reformatting arrays so that they would be 2-dimensional
    if len(X.shape) == 1:
        X = X.reshape(-1,1) # N_trials X 1
    if len(T.shape) == 1:
        T = T.reshape(-1,1) # N_trials X 1
    if len(NT.shape) == 1:
        NT = NT.reshape(-1,1) # N_trials X N_non-targets

    if ((X.shape[1] > 1 or T.shape[1] > 1 or X.shape[0] != T.shape[0])
        or (~np.isnan(NT).any()
            and (NT.shape[0] != X.shape[0] or NT.shape[0] != T.shape[0]))):
        print('Input is not correctly dimensioned')
        B = np.nan
        LL = np.nan

    else:
        LL = np.inf
        K = 0 + 10 * np.random.random(iter)
        M = -np.pi + 2*np.pi*np.random.random(iter)
        Pt = np.random.random(iter)
        Pn = (1 - Pt) * np.random.random(iter)

        startB = np.vstack([K, M, Pt, Pn]).transpose()
        boundK = [0, None]
        boundM = [-np.pi, np.pi]
        boundPT = [0, 1]
        boundPN = [0, 1]

        for i in range(iter):
            start = startB[i]

            fit = optimize.minimize(llhood, start,
                                    args=(X, T, NT),
                                    bounds = (boundK,
                                              boundM,
                                              boundPT,
                                              boundPN))
            if fit['success'] and fit['fun'] < LL:
                LL = fit['fun']
                B = fit['x']

        B = np.append(B, 1 - B[2:].sum())

    return (B, LL)
