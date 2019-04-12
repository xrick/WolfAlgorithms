# -*- coding: utf-8 -*-
import numpy as np
import json

'''
Homework3 of Speech Recognition Course 2019 by Berlin Chen
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
    alpha = np.zeros((T,N),dtype=float) #means alpha is a TxN matrix
    alpha[0,:] = pi * b[:,o[0]]
    for t in range(1,T):
        for j in range(N):
            #print("current t is {0} and j is {1}".format(t,j))
            #以下做內積（也許可以簡化掉這一個迴圈，用np.dot）
            '''
            p = 0
            for idx in range(N):
                p+=alpha[t-1,idx]*a[idx,j]
            alpha[t,j] = p * b[j,o[t]]
            '''
            alpha[t,j] = np.dot(alpha[t-1,:],a[:,j])*b[j,o[t]] #使用np.dot做內積
    total_p = 0
    for k in range(N):
        total_p += alpha[T-1][k]

    return alpha, total_p

''' Final Result is as following:
total p is 0.0004967268975999999
The alpha is 
[[3.50000000e-01 2.00000000e-02 9.00000000e-02]
 [1.79200000e-01 8.50000000e-03 3.57000000e-02]
 [2.52100000e-02 1.25880000e-02 2.21560000e-02]
 [3.02824000e-03 6.62040000e-03 5.59128000e-03]
 [1.47273120e-03 9.45268800e-04 1.89014720e-03]
 [2.11233200e-04 4.60284960e-04 4.28602080e-04]
 [3.69826262e-04 2.23192336e-05 1.04581402e-04]]
'''
   

def Backward(a,b,o,pi):
    N = np.shape(b)[0]
    T = np.shape(o)[0]
    beta = np.zeros((T,N))
    beta[T-1,:] = 1.0
    for t in reversed(range(T-1)): #python tips: range(T)
        for i in range(N):
            '''
            p = 0
            for j in range(N):
                p += a[i,j] * b[j,o[t+1]] * beta[t+1,j]
            '''
            beta[t, i] = np.sum(a[i,:]*b[:,o[t+1]]*beta[t+1,:])
    total_p2 = 0
    for k in range(N):
        total_p2 += pi[k] * b[k,o[0]] * beta[0,k]
    
    return beta, total_p2

'''
total p2 is 0.0004967268976
The beta is 
[[1.10357056e-03 9.75693700e-04 1.01070364e-03]
 [2.15212400e-03 2.27718200e-03 2.56891400e-03]
 [7.70388000e-03 9.30452000e-03 8.36734000e-03]
 [2.99320000e-02 3.16660000e-02 3.51340000e-02]
 [1.09200000e-01 1.30600000e-01 1.12400000e-01]
 [5.00000000e-01 4.40000000e-01 4.40000000e-01]
 [1.00000000e+00 1.00000000e+00 1.00000000e+00]]
'''


'''
######################################################################
Problem 2 :
Given a HMM, M, and a sequence of observations, x
Find the sequence Q of hidden states that maximizes P(x, Q|M)
######################################################################
Note : the [up,down,unchanged] is encoded to [0,1,2]
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
    ans is q_ = [0. 0. 2. 2. 2. 2. 0.] 
   delta is 
[[3.50000e-01 2.00000e-02 9.00000e-02]
 [1.47000e-01 7.00000e-03 2.10000e-02]
 [1.76400e-02 8.82000e-03 1.76400e-02]
 [1.05840e-03 2.11680e-03 2.64600e-03]
 [2.11680e-04 1.90512e-04 7.93800e-04]
 [3.17520e-05 4.76280e-05 1.19070e-04]
 [3.33396e-05 1.42884e-06 1.78605e-05]]
psi_ is 
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 2.]
 [1. 1. 2.]
 [2. 2. 2.]
 [2. 1. 2.]]

    '''
            

    

'''
######################################################################
Problem 3 :
Given an unknown HMM, M, and a sequence of observations, x
Find parameters θ that maximize P(x|θ, M)
######################################################################
'''
#This BaumWelch is according to the paper : 
def BaumWelch(a,b,o,pi,criterion=0.001):
    T = np.shape(o)[0]
    N = np.shape(b)[0]
    while True:
        # Initialize alpha
        alpha = Forward(a,b,o,pi)
         # Initialize beta
        beta = Backward(a,b,o,pi)
        xi = np.zeros((T-1,N,N),dtype=float)
        for t in range(T-1):
            denominator = np.sum( np.dot(alpha[t,:],a) * b[:,o[t+1]] * beta[t+1,:])
            for i in range(N):
                molecular = alpha[t,i] * a[i,:] * b[:,o[t+1]] * beta[t+1,:]
                xi[t,i,:] = molecular / denominator
        gamma = np.sum(xi,axis=2)
        prod = (alpha[T-1,:] * beta[T-1,:])
        gamma = np.vstack((gamma, prod /np.sum(prod)))
        newpi = gamma[0,:]
        newA = np.sum(xi,axis=0) / np.sum(gamma[:-1,:],axis=0).reshape(-1,1)
        newB = np.zeros(b.shape,dtype=float)
        for k in range(b.shape[1]):
            mask = o == k
            newB[:,k] = np.sum(gamma[mask,:],axis=0) / np.sum(gamma,axis=0)
        if np.max(abs(pi - newpi)) < criterion and np.max(abs(a - newA)) < criterion and np.max(abs(b - newB)) < criterion:
            break

        return newA,newB,newpi

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
        [0.3, 0.3, 0.4]]
    )
    pi = [0.5, 0.2, 0.3]
    O = [0, 0, 2, 1, 2, 1, 0]
       # ['up', 'up', 'unchanged', 'down', 'unchanged', 'down', 'up']
    ####################################################
    #alpha_, p_ = Forward(A,B,O,pi)
    #print("The alpha is {} and total p is {}".format(alpha_,p_))
    #beta_, p2_ = Backward(A,B,O,pi)
    #print("The beta is {} and total p2 is {}".format(beta_,p2_))
    #beta_ = Backward(A,B,O,pi)
    #print("p_ is {0}".format(p_))
    #print("alpha_ is {}".format(alpha_))
    #print("beta_ is {0}".format(beta_))
    #delta_, psi_ ,q_= Viterbi(A,B,O,pi)
    #print("delta is {}".format(delta_))
    #print("psi_ is {0}".format(psi_))
    #print("q_ is {}".format(q_))
    _newA, _newB, newPi = BaumWelch(A,B,O,pi)

def test_fun():
    i = 8
    j = 6
    m = 0
    m = i==j
    print("current m is {}".format(m))

if __name__ == "__main__":
    run_app()
    #test_fun()

