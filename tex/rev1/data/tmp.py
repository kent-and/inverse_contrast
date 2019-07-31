
from numpy import * 
a = loadtxt("tmp.txt")
(alpha,beta,gamma,k,noise,iteration, D1_rel, D2_rel, Boundary_1, Boundary_2) = [a[:,i] for i in range(0,10)]   

print (alpha)
data= {} 
for i in range(len(alpha)): 
  data [(alpha[i], beta[i], gamma[i], k[i], noise[i])] = (D1_rel, D2_rel, Boundary_1, Boundary_2)  


alphas = unique(alpha) 
betas = unique(beta) 
gammas = unique(gamma) 
ks = unique(k) 
noises = unique(noise)

print "alphas ", alphas  
print "betas ", betas


for a in alphas: 
  for b in betas: 
    for g in gammas: 
      for k in ks: 
        for n in noises: 
          if not data.has_key((a,b,g,k,n)): 
            print "%2.1e %2.1e %2.1e %2d %2.1e " % (a, b, g, k, n)  


