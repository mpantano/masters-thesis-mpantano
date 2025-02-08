## Transitional MCMC subroutine
## Description: Computes model evidence and produces stationary samples from multimodal/NonGaussian pdfs
## Author: Rimple Sandhu; Email: rimple_sandhu@outlook.com

#!/usr/bin/python
import os, math, sys, random
import numpy as np
import numpy.linalg as la
import scipy.stats as st

def TMCMC(Fx,Np,x_low,x_up,Nj):
# Usage information:
##### Input:
# Fx: posterior pdf/likelihood function
# Np: Dimension of posterior pdf
# x_low: Lower bound of x
# x_up: Upper bound of x
# Nj: Number of TMCMC sampler per stage
# beta: Scaling parameter for proposal pdf (2.4/sqrt(Np))
##### Output:
# x (Np,Nj): Stationary MCMC samples
# ll(Nj,1): pdf ordinate
# lsj(1,1): Log-evidence
# p : pdf scaling  


# Obtain prior samples and compute log-prior
    x = np.zeros((Np,Nj));
    lprior = np.zeros((Np,1));
    for i in range(0,Np):
        x[i,:] = x_low[i] + (x_up[i]-x_low[i])*np.random.random_sample(Nj)
        lprior[i] = -np.log(x_up[i]-x_low[i])
    y = x
# define Log-Likelihood function
    ll = np.asarray([0.0]*Nj)
    def F(xin):
        return -np.sum(lprior)+ Fx(xin)
    for i in range(0,Nj):
        ll[i] = F(x[:,i])

# decide the stages for TMCMC, if fixed
    stage_flag = 0
    Nm = 15 # number of stages
    if stage_flag == 1:
        p = np.logspace(-4,0,Nm) # log scale
    elif stage_flag == 2:
        p = np.linspace(0.0001,1,Nm) # linear scale
    else:
        p = []
    print('Stage Flag',stage_flag)
# TMCMC sampling starts
    beta = 0.3#2.4/np.sqrt(Np)
    p_prv = 0.0
    p_cur = 0.0
    lsj   = 0.0
    j  =  1
    print('TMCMC sampling:')
    while (p_cur < 1):
        if stage_flag == 0:
            p_low = p_prv
            p_up = 1.0
            while (p_up - p_low > 1e-06):
                p_cur = (p_low + p_up)/2.0
                temp = np.exp((p_cur-p_prv)*(ll-np.max(ll)))
                cov_w = np.std(temp)/np.mean(temp)
                if cov_w > 1.0:
                    p_up = p_cur
                else:
                    p_low = p_cur
            if abs(1.0 - p_cur) < 1e-02:
                p_cur= 1.0
            p.append(p_cur)
        else:
            p_cur = p[j]
        w = np.exp((p_cur-p_prv)*(ll-np.max(ll)))
        w_norm = w/np.sum(w)
        lsj = lsj + np.log(np.mean(w)) + (p_cur-p_prv)*np.max(ll)

# Compute proposal covariance for MH
        x_mu = x@w_norm
        #print(x_mu)
        mhsigma  = np.zeros((Np,Np))
        for i in range(0,Nj):
            mhsigma= mhsigma + beta**2*w_norm[i]*np.outer(x[:,i]-x_mu,x[:,i]-x_mu) 
        mhSigma_chol = la.cholesky(mhsigma);

# Print output to screen
        print('--- Stage ',j, ': ',p_cur)
        #print('P_j = %8.3f ; P(D|M) = %7.4E \n',p_cur, exp(lsj))

# MCMC step
        I_j = np.random.choice(Nj,Nj,p=w_norm)
        ll_prv = ll
        x_prv = x
        rej_cur = 0
        for k in range(0,Nj):
            I_jk = I_j[k]
            x_new = x_prv[:,I_jk] + mhSigma_chol@np.random.normal(0,1,Np)
            ll_new = F(x_new);
            ratio = np.exp(p_cur*(ll_new - ll_prv[I_jk]))
            if ratio > np.random.random_sample() and np.sum(x_new < x_up)== Np and np.sum(x_new > x_low) == Np:
                x[:,k] = x_new
                ll[k] = ll_new
                x_prv[:,I_jk] = x_new
                ll_prv[I_jk] = ll_new
            else:
                x[:,k] = x_prv[:,I_jk]
                ll[k] = ll_prv[I_jk]
                rej_cur = rej_cur + 1
        #chst[j,2] = rej_cur*100/Nj
# proceed to next stage
        y = np.concatenate((y,x),axis=1)  
        p_prv = p_cur
        j = j+1
    print('Log-evidence: ',lsj)
    return x,y,ll,lsj,p

