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
    alpha = np.zeros((T,N)) #means alpha is a NxT matrix, it represents a trellis 
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
    return alpha

''' Final Result is as following:
[[3.50000000e-01 2.00000000e-02 9.00000000e-02]
 [1.79200000e-01 8.50000000e-03 3.57000000e-02]
 [2.52100000e-02 1.25880000e-02 3.32340000e-02]
 [3.47136000e-03 7.28508000e-03 7.25298000e-03]
 [1.72530960e-03 1.08152820e-03 3.46666680e-03]
 [2.96261658e-04 6.09712236e-04 6.88410288e-04]
 [5.30584060e-04 3.11007031e-05 1.57619977e-04]]
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

'''
[[3.50000000e-01 1.79200000e-01 2.52100000e-02 3.47136000e-03 1.72530960e-03 2.96261658e-04 5.30584060e-04]
 [2.00000000e-02 8.50000000e-03 1.25880000e-02 7.28508000e-03 1.08152820e-03 6.09712236e-04 3.11007031e-05]
 [9.00000000e-02 3.57000000e-02 3.32340000e-02 7.25298000e-03 3.46666680e-03 6.88410288e-04 1.57619977e-04]]
'''
   

def Backward(a,b,o,pi):
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
beta_ is 
[[0.00158631 0.00140636 0.00151076]
 [0.00302364 0.00317028 0.00421628]
 [0.00918756 0.011013   0.01050294]
 [0.034428   0.036162   0.046374  ]
 [0.1092     0.1306     0.1124    ]
 [0.5        0.44       0.44      ]
 [1.         1.         1.        ]]
'''


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
    q = np.zeros(T)
    for t in range(T):
        for j in range(N):
            if t == 0:
                delta[t,j] = pi[j]*b[j,o[t]] #initalize the delta's 
                print("delta[{},{}] is {}".format(t, j, delta[t, j]))
            else:
                p=-1e9
                for i in range(N):
                    w = delta[t-1,i]*a[i,j]
                    if w>p:
                        p=w
                        psi[t,j] = i
                delta[t,j] = p * b[j,o[t]]
    
    p2 = -1e9
    for j in range(N):
        if delta[T-1,j] > p2:
            p2 = delta[T-1,j]
            q[T-1] = j
    for t in reversed(range(T)):
        if t-1 > -1:
            q[t-1] = psi[t,int(q[t])]
    return delta, psi, q

    '''
   delta is [[3.50000e-01 2.00000e-02 9.00000e-02]
 [1.47000e-01 7.00000e-03 2.10000e-02]
 [1.76400e-02 8.82000e-03 1.76400e-02]
 [1.05840e-03 2.11680e-03 2.64600e-03]
 [2.11680e-04 1.90512e-04 7.93800e-04]
 [3.17520e-05 4.76280e-05 1.19070e-04]
 [3.33396e-05 1.42884e-06 1.78605e-05]]
psi_ is [[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 2.]
 [1. 1. 2.]
 [2. 2. 2.]
 [2. 1. 2.]]
q_ is [0. 0. 2. 2. 2. 2. 0.]
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
        pass


'''
O mean a set of observation sequence.in our the homework the O is a vector of observed steps.
#up:0 , down:1, unchanged:2
#up, up, unchanged, down, unchanged, down, up 
'''
def run_app():

    ####################### HMM ########################
    # get the transition matrix A
    A = np.array([
    [0.6, 0.2, 0.2],
    [0.5, 0.3, 0.2],
    [0.4, 0.1, 0.5]]
    )

    B = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.6]]
    )
    pi = [0.5, 0.2, 0.3]
    O = [0, 0, 2, 1, 2, 1, 0]
    ####################################################
    #alpha_ = Forward2(A,B,O,pi)
    #beta_ = Backward(A,B,O,pi)
    #print("p_ is {0}".format(p_))
    #print("alpha_ is {}".format(alpha_))
    #print("beta_ is {0}".format(beta_))
    delta_, psi_ ,q_= Viterbi(A,B,O,pi)
    print("delta is {}".format(delta_))
    print("psi_ is {0}".format(psi_))
    print("q_ is {}".format(q_))


if __name__ == "__main__":
    run_app()
    #test_fun()


