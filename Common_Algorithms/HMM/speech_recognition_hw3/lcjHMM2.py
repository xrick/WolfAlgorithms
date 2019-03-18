# -*- coding: utf-8 -*-
import numpy as np
import json

'''
Homework2,3 of Speech Recognition Course 2019 by Berlin Chen
Author : Rick Liao
'''
'''
######################################################################
Problem 1 :Given a HMM, M, and a sequence of observations, xFind P(x|M)
######################################################################
'''

#以列為主的算法
def Forward(a,b,o,pi):
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    alpha = np.zeros((N,T)) #means alpha is a NxT matrix, it represents a trellis 
    for t in range(T):
        for j in range(N):
            print("current t is {0} and j is {1}".format(t,j))
            if t==0 :
                alpha[t,j] = pi[j]*b[j,o[t]]
            else:
                p = 0
                for idx in range(N):
                    print("inner i is {0}".format(idx))
                    p+=alpha[t-1,idx]*a[idx,j]
                alpha[t,j] = p * b[j,o[t]]
    p = 0
    for i in range(N):
        p+=alpha[T-1,i]
    return p
'''
tolerance = 1e-6
    scaling = True
    max_iterations = 10000
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    
    alpha = np.zeros((N,T))
    # initialise first column with observation values
    alpha[:,0] = pi*b[:,o[0]] 
    c = np.ones((T))
        
    if scaling:
        c[0]=1.0/np.sum(alpha[:,0])
        alpha[:,0]=alpha[:,0]*c[0]
        for t in range(1,T):
            c[t]=0
            for i in range(N):
                alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
            c[t]=1.0/np.sum(alpha[:,t])
            alpha[:,t]=alpha[:,t]*c[t]
    else:
        for t in range(1,T):
            for i in range(N):
                alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
    return alpha, c
'''
#以行為主的算法
def Forward2(a,b,o,pi):
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    alpha = np.zeros((N,T))
    alpha[:,0] = pi*b[:,o[0]]
    print("alpha[:,0] is {0}".format(alpha[:, 0]))
    for t in range(1, T):
            for i in range(N):
                #b[i, o[t]]第i行
                #a[:,i] 第i行
                alpha[i, t] = b[i, o[t]] * np.sum(alpha[:, t-1] * a[:, i])
            print("current alpha is {0}".format(alpha))
    return alpha
   

def Backward(a,b,o,pi):
    '''
    double backward(int* o, int T)
    {
    for (int t=T-1; t>=0; --t)
        for (int i=0; i<N; ++i)
            if (t == T-1)
                β[t][i] = 1.0;
            else
            {
                double p = 0;
                for (int j=0; j<N; ++j)
                    p += a[i][j] * b[j][o[t+1]] * β[t+1][j];
                β[t][i] = p;
            }
 
    double p = 0;
    for (int j=0; j<N; ++j)
        p += π[j] * b[j][o[0]] * β[0][j];
    return p;
    }
    '''
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    beta = np.zeros((T,N))
    for t in reversed(range(T)): #python tips: range(T)
        for i in range(N):
            if t==T-1:
                beta[t,i] = 1.0 #將T-1列，也是最後一行，設為1.0
            else:
                p = 0
                for j in range(N):
                       # a[i,j]:a的第i列
                       # b[j,o[t+1]]:b的第j行
                       # beta的第t+1列
                    p += a[i,j] * b[j,o[t+1]] * beta[t+1,j]
                beta[t, i] = p  # 將第t列，設為p值

    p2 = 0
    for k in range(N):
        p2 += pi[k] * b[k,o[0]] * beta[0,j]
    return beta


'''
######################################################################
Problem 2 :
Given a HMM, M, and a sequence of observations, x
Find the sequence Q of hidden states that maximizes P(x, Q|M)
######################################################################
'''

def Viterbi(a,b,o,pi):
    T = np.shape(o)[0]
    N = np.shape(b)[0]
    delta = np.zeros((T,N))
    psi=np.zeros((T,N))
    q = np.zeros(N)
    for t in range(T):
        for j in range(N):
            if t == 0:
                delta[t,j] = pi[j]*b[j,o[t]]
            else:
                p=-1e9
                for i in (N):
                    w = delta[t-1,t]*a[i,j]
                    if w>p:
                        p=w
                        psi[t,j] = i
                delta[t,j] = p * b[j,o[t]]

    p2 = -1e9
    for j in (N):
        if delta[T-1,j] > p:
            p2 = delta[T-1,j]
            q[T-1] = j
    for t in reversed(range(T)):
        q[t-1] = psi[t,q[t]]

    return delta, psi, q
            

    '''
def HMMViterbi(a, b, o, pi):
# Implements HMM Viterbi algorithm            
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    
    path = np.zeros(T)
    delta = np.zeros((N,T))
    phi = np.zeros((N,T))
    
    delta[:,0] = pi * b[:,o[0]]
    phi[:,0] = 0
    
    for t in range(1,T):
        for i in range(N):
            delta[i,t] = np.max(delta[:,t-1]*a[:,i])*b[i,o[t]]
            phi[i,t] = np.argmax(delta[:,t-1]*a[:,i])
    
    path[T-1] = np.argmax(delta[:,T-1])
    for t in range(T-2,-1,-1):
        path[t] = phi[int(path[t+1]),t+1]
    
    return path,delta, phi
    '''


'''
######################################################################
Problem 3 :
Given an unknown HMM, M, and a sequence of observations, x
Find parameters θ that maximize P(x|θ, M)
######################################################################
'''
def BaumWelch(a,b,o,pi):
    alpha_ = Forward2(a,b,o,pi)
    beta_ = Backward(a,b,o,pi)
    T = np.shape(o)[0]
    N = np.shape(b)[0]
    gamma = np.zeros((T,N))
    epslon = np.zeros((N,N))

    for t in range(T):
        p = 0
        for i in range(N):
            p += alpha_[t,i] * beta_[t,i]
        assert(p!=0)

        for i in range(N):
            gamma[t,i] = alpha_[t,i] * beta_[t,i] / p

    

    '''
tolerance = 1e-6
    scaling = True
    max_iter = 10000
    # Implements HMM Baum-Welch algorithm        
        
    T = np.shape(o)[0]

    M = int(max(o))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

    digamma = np.zeros((N,N,T))

    
    # Initialise A, B and pi randomly, but so that they sum to one
    np.random.seed(rand_seed)
        
    # Initialisation can be done either using dirichlet distribution (all randoms sum to one) 
    # or using approximates uniforms from matrix sizes
    if dirichlet:
        pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))    
        a = np.random.dirichlet(np.ones(N),size=N)    
        b=np.random.dirichlet(np.ones(M),size=N)
    else:    
        pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
        pi=1.0/N*np.ones(N)-pi_randomizer
        a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
        a=1.0/N*np.ones([N,N])-a_randomizer
        b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
        b = 1.0/M*np.ones([N,M])-b_randomizer
    error = tolerance+10
    itter = 0
    while ((error > tolerance) & (itter < max_iter)):   
        prev_a = a.copy()
        prev_b = b.copy()
        # Estimate model parameters
        alpha, c = HMMfwd(a, b, o, pi)
        beta = HMMbwd(a, b, o, c) 
    
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
            digamma[:,:,t] /= np.sum(digamma[:,:,t])

        for i in range(N):
            for j in range(N):
                digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
        digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
        # Maximize parameter expectation
        for i in range(N):
            pi[i] = np.sum(digamma[i,:,0])
            for j in range(N):
                a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
    	

            for k in range(M):
                filter_vals = (o==k).nonzero()
                b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
        error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
        itter += 1            
            
        if verbose:            
            print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
    return a, b, pi, alpha
    '''

'''
O mean a set of observation sequence.in our the homework the O is a vector of observed steps.
#up:0 , down:1, unchanged:2
#up, up, unchanged, down, unchanged, down, up 
'''
def run_app():

    ####################### HMM ########################
    # get the transition matrix A
    A = np.transpose(np.array([
    [0.6, 0.2, 0.2],
    [0.5, 0.3, 0.2],
    [0.4, 0.1, 0.5]]
    ))

    B = np.transpose(np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.6]]
    ))
    pi = [0.5, 0.2, 0.3]
    O = [0, 0, 2, 1, 2, 1, 0]
    ####################################################
    beta_, p_ = Backward(A,B,O,pi)
    print("p_ is {0}".format(p_))
    print("beta_ is {0}".format(beta_))

def test_fun():
    A = np.transpose(np.array([
        [0.6, 0.2, 0.2],
        [0.5, 0.3, 0.2],
        [0.4, 0.1, 0.5]]
    ))

    B = np.transpose(np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.6]]
    ))

if __name__ == "__main__":
    run_app()
    #test_fun()

'''
##########################################################
                 Forward-Backward Algorithm
##########################################################
const int N = 3, M = 3, T = 15;  
double π[N], a[N][N], b[N][M];  // HMM  
double α[T][N]; // 可以簡化成α[2][N]  
double β[T][N]; // 可以簡化成β[2][N]  
  
double forward(int* o, int T)  
{  
    for (int t=0; t
        for (int j=0; j
            if (t == 0)  
                α[t][j] = π[j] * b[j][o[t]];  
            else  
            {  
                double p = 0;  
                for (int i=0; i
                    p += α[t-1][i] * a[i][j];  
                α[t][j] = p * b[j][o[t]];  
            }  
  
    double p = 0;  
    for (int i=0; i
        p += α[T-1][i];  
    return p;  
}  
  
double backward(int* o, int T)  
{  
    for (int t=T-1; t>=0; --t)  
        for (int i=0; i
            if (t == T-1)  
                β[t][i] = 1.0;  
            else  
            {  
                double p = 0;  
                for (int j=0; j
                    p += a[i][j] * b[j][o[t+1]] * β[t+1][j];  
                β[t][i] = p;  
            }  
  
    double p = 0;  
    for (int j=0; j
        p += π[j] * b[j][o[0]] * β[0][j];  
    return p;  

} 
##########################################################
                 Viterbi Algorithm
##########################################################
//Viterbi Algorithm
const int N = 3, M = 3, T = 15;  
double π[N], a[N][N], b[N][M];  // HMM  
double δ[T][N]; // 可以簡化成δ[2][N]  
int ψ[T][N];  
  
double decode(int* o, int T, int* q)  
{  
    for (int t=0; t
        for (int j=0; j
            if (t == 0)  
                δ[t][j] = π[j] * b[j][o[t]];  
            else  
            {  
                double p = -1e9;  
                for (int i=0; i
                {  
                    double w = δ[t-1][i] * a[i][j];  
                    if (w > v) p = w, ψ[t][j] = i;  
                }  
                δ[t][j] = p * b[j][o[t]];  
            }  
  
    double p = -1e9;  
    for (int j=0; j
        if (δ[T-1][j] > v)  
            p = δ[T-1][j], q[T-1] = j;  
  
    for (int t=T-1; t>0; --t)  
        q[t-1] = ψ[t][q[t]];  
  
    return p;  
} 

##########################################################
                 Baum-Welch Algorithm
##########################################################

const int N = 3, M = 3, T = 15;  
double π[N], a[N][N], b[N][M];  // HMM  
double α[T][N], β[T][N];        // evaluation problem  
double δ[T][N]; int ψ[T][N];    // decoding problem  
double γ[T][N], ξ[T][N][N];     // learning problem  
  
void learn(int* o, int T)  
{  
    forward(o, T);  
    backward(o, T);  
  
    for (int t=0; t
    {  
        double p = 0;  
        for (int i=0; i
            p += α[t][i] * β[t][i];  
        assert(p != 0);  
  
        for (int i=0; i
            γ[t][i] = α[t][i] * β[t][i] / p;  
    }  
  
    for (int t=0; t1; ++t)  
    {  
        double p = 0;  
        for (int i=0; i
            for (int j=0; j
                p += α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j];  
        assert(p != 0);  
  
        for (int i=0; i
            for (int j=0; j
                ξ[t][i][j] = α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j] / p;  
    }  
  
    // 更新Π  
    for (int i=0; i
        π[i] = γ[0][i];  
  
    // 更新A  
    for (int i=0; i
    {  
        double p2 = 0;  
        for (int t=0; t1; ++t)  
            p2 += γ[t][i];  
        assert(p2 != 0);  
  
        for (int j=0; j
        {  
            double p1 = 0;  
            for (int t=0; t1; ++t)  
                p1 += ξ[t][i][j];  
            a[i][j] = p1 / p2;  
        }  
    }  
  
    // 更新B  
    for (int i=0; i
    {  
        double p[M] = {0}, p2 = 0;  
        for (int t=0; t
        {  
            p[o[t]] += γ[t][i];  
            p2 += γ[t][i];  
        }  
        assert(p2 != 0);  
  
        for (int k=0; k
            b[i][k] = p[k] / p2;  
    }  
}  

'''

'''
[3.50000000e-01 1.58200000e-01 3.16980000e-02 3.44142000e-03 2.56616100e-03 3.28743684e-04 5.93456583e-04]
'''
