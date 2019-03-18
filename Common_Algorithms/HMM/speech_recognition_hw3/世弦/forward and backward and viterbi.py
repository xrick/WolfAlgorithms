pi={1:0.5, 2:0.2, 3:0.3}
a={1:{1:0.6, 2:0.2, 3:0.2}, 
   2:{1:0.5, 2:0.3, 3:0.2},
   3:{1:0.4, 2:0.1, 3:0.5}}
b={1:{'up':0.7, 'down':0.1, 'unchanged':0.2}, 
   2:{'up':0.1, 'down':0.6, 'unchanged':0.3}, 
   3:{'up':0.3, 'down':0.3, 'unchanged':0.4}}

O=['','up', 'up', 'unchanged', 'down', 'unchanged', 'down', 'up']
T=len(O)-1
N=len(a)

#forward algorithm 計算 P(O|HMM)
alpha={1:{}}
for i in range(1, N+1):
    alpha[1][i]=pi[i]*b[i][O[1]]
    
for t in range(1, T):
    alpha[t+1]={}
    for j in range(1, N+1):
        tmp=0
        for i in range(1, N+1):
            tmp+=alpha[t][i]*a[i][j]
        alpha[t+1][j]=tmp*b[j][O[t+1]]
        
forward_ans=0                   
for i in range(1, N+1):
    forward_ans+=alpha[T][i]
print ('Forward alg. P(O|HMM):', forward_ans)        #P(O|HMM)=0.0004967268976

#backward algorithm 計算 P(O|HMM)
beta={T:{}}
for i in range(1, N+1):
    beta[T][i]=1
    
for t in range(T-1, 0, -1):
    beta[t]={}
    for i in range(1, N+1):
        tmp=0
        for j in range(1, N+1):
            tmp+=a[i][j]*b[j][O[t+1]]*beta[t+1][j]
        beta[t][i]=tmp
        
backword_ans=0
for j in range(1, N+1):
    backword_ans+=pi[j]*b[j][O[1]]*beta[1][j]
print ('Backward alg. P(O|HMM):', backword_ans)      #P(O|HMM)=0.0004967268976

#summation 1~N alpha*beta
ans=0
for i in range(1, N+1):
    ans+=alpha[3][i]*beta[3][i]
print ('Forward*backward P(O|HMM):', ans)            #P(O|HMM)=0.0004967268976

#viterbi alg. deconding
delta={1:{}}
psi={}
for i in range(1, N+1):
    delta[1][i]=pi[i]*b[i][O[1]]
    
for t in range(1, T):
    delta[t+1]={}
    psi[t+1]={}
    for j in range(1, N+1):
        tmp=[]
        for i in range(1, N+1):
            tmp.append(delta[t][i]*a[i][j])
        delta[t+1][j]=max(tmp)*b[j][O[t+1]]
        psi[t+1][j]=tmp.index(max(tmp))+1

best_state_seq=[0]*(T+1)                            #backtrace
sT=list(delta[7].values())
sT=sT.index(max(sT))+1
best_state_seq[T]=sT
for t in range(T-1, 0, -1):
    best_state_seq[t]=psi[t+1][best_state_seq[t+1]]
print ('best state sequence:', best_state_seq[1:])  #[1, 1, 3, 3, 3, 3, 1]

            


