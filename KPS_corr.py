import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2
import os
import sys

import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2
import os
import sys

scriptpath = sys.path[0]
R_xlsx = pd.read_excel(os.path.join(scriptpath, 'Correlations.xlsx'), index_col=0, header=0)
R = np.array(R_xlsx)

p = 5
k = 9
n = 1
alpha = 0.05

R_cal = np.empty([p ** 2, k ** 2]) ## equation (5)
for i in range(1,p+1):
    for j in range(1,p+1):
        block_ij = R[((i-1)*k):(i*k), ((j-1)*k):(j*k)] # k*k     
        vector_ij = block_ij.flatten(order='F') # k^2*1
        R_cal[(j-1)*p+i-1,:] = vector_ij # the p-th row
R_cal = R_cal
vec_R_cal = R_cal.flatten(order="F")

V = np.identity(np.outer(vec_R_cal.T,vec_R_cal.T).shape[0])

L, Sigma_v, Nt = np.linalg.svd(R_cal) ## equation (9)
# L, Sigma_v, Nt = scipy.linalg.svd(R_cal,lapack_driver='gesvd') ## alternative method
Sigma = np.diag(Sigma_v)
N = Nt.T

length = max(np.shape(L),np.shape(N))[0]
width = min(np.shape(L),np.shape(N))[0]
diff = length - width 

happend = np.zeros([np.shape(Sigma)[0], diff])
vappend = np.zeros([diff, np.shape(Sigma)[0]])

if np.shape(L)[0] > np.shape(N)[0]:
    Sigma = np.vstack((Sigma, vappend))
elif np.shape(L)[0] < np.shape(N)[0]:
    Sigma = np.hstack((Sigma, happend))
  
## Components of the KPST (the elements are only identified up to scale, so we normalize the first diagonal element to 1, see theorem 1)
L2 = np.delete(L, 0, 1)/L[0,0]
Sigma2 = np.delete(np.delete(Sigma, 0, 1), 0, 0)/Sigma[0,0]
N2 = np.delete(N, 0, 1)/N[0,0]

## equation (22)
K1 = Sigma2.flatten(order='F') 
K2 = np.kron(N2,L2).T @ V @ np.kron(N2,L2)
KPST = n* K1.T @ np.linalg.inv(K2) @ K1

## degree of freedom
dof = (0.5*k*(k+1)-1)*(0.5*p*(p+1)-1)

## Test
if KPST < chi2.ppf(1-.05, dof):
    print("KPST = ",KPST," chi2(df,0.05) = ",chi2.ppf(1-alpha, dof)," \n We can't reject the null that R has KPS at nominal size 0.05. \n  We conclude that R has KPS.")
else: 
    print("KPST = ",KPST," chi2(df,0.05) = ",chi2.ppf(1-alpha, dof),"\n We reject the null that R has KPS at nominal size 0.05.\n We conclude that R doesn't has KPS.")