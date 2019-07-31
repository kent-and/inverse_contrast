
from numpy import * 
a = loadtxt("tmp.txt")
(alpha,beta,gamma,k,noise,iteration, D1_rel, D2_rel, Boundary_1, Boundary_2) = [a[:,i] for i in range(0,10)]   

print (alpha)
data= {} 
for i in range(len(alpha)): 
  data [(alpha[i], beta[i], gamma[i], k[i], noise[i])] = (D1_rel[i], D2_rel[i], Boundary_1[i], Boundary_2[i])  


alphas = unique(alpha) 
betas = unique(beta) 
gammas = unique(gamma) 
ks = unique(k) 
noises = unique(noise)

print "all alpha ", alpha.shape
print "alphas ", alphas
print "betas ", betas


counters = {} 
ncounters = {} 
for a in alphas: 
  for b in betas: 
    for g in gammas: 
      for k in ks: 
        for n in noises: 
          if data.has_key((a,b,g,k,n)): 
            D1 = data[(a,b,g,k,n)][0]
            if D1 < 0.1: 
              if counters.has_key((a,b,g)): counters[(a,b,g)] += 1 
              else: counters[(a,b,g)] = 1; ncounters[(a,b,g)] = 0  
            else: 
              if ncounters.has_key((a,b,g)): ncounters[(a,b,g)] += 1 
              else: counters[(a,b,g)] = 0; ncounters[(a,b,g)] = 1 


for a in alphas: 
  for b in betas: 
    for g in gammas: 
      if counters.has_key((a,b,g)): print (a, b, g, "below 0.1 ", counters[(a,b,g)], " total ", counters[(a,b,g)] + ncounters[(a,b,g)]) 

