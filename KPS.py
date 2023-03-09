import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2
import os
import sys

## Import region

scriptpath = sys.path[0]

region1 = pd.read_excel(os.path.join(scriptpath, 'test.xlsx'), sheet_name="Reg=1", index_col=0, header=0)
region1['Time'] = region1.reset_index().index+1

k = 3
p = len(region1.columns) - 1
n = len(region1) ## before melting
alpha = 0.05

print("[p k n] = ",p,k,n)

region1 = pd.melt(region1, id_vars = "Time", value_vars = range(1,8), var_name = "Industry")
region1["Region"] = 1

region2 = pd.read_excel(os.path.join(scriptpath, 'test.xlsx'), sheet_name="Reg=1", index_col=0, header=0)
region2['Time'] = region2.reset_index().index+1
region2 = pd.melt(region2, id_vars = "Time", value_vars = range(1,8), var_name = "Industry")
region2["Region"] = 2

region3 = pd.read_excel(os.path.join(scriptpath, 'test.xlsx'), sheet_name="Reg=1", index_col=0, header=0)
region3['Time'] = region3.reset_index().index+1
region3 = pd.melt(region3, id_vars = "Time", value_vars = range(1,8), var_name = "Industry")
region3["Region"] = 3

longdata = pd.concat([region1,region2,region3])
longdata = longdata.reindex(sorted(longdata.columns), axis=1)

print("data = \n", longdata)

T_list = []
for t in longdata["Time"].unique():
    T_tmp = longdata.loc[longdata["Time"] == t]["value"]
    # globals()['T' + str(t)] = T_tmp
    T_list.append(T_tmp)

n = len(T_list)
R_sum = 0
R_list = []
count = 1
for fi in T_list:
    Ri = np.outer(fi,fi)
    R_list.append(Ri)
    # globals()['R' + str(count)] = Ri
    R_sum += Ri ## pk*pk
    count +=1
R = R_sum/n ## pk*pk, equation (1)
R_list.append(R)

vec_R_cal_n = np.empty((68, 441))
for t in range(len(R_list)):
    R_tmp = R_list[t]
    R_cal_tmp = np.empty([p ** 2, k ** 2]) ## equation (5)
    for i in range(1,p+1):
        for j in range(1,p+1):
            block_ij = R_tmp[((i-1)*k):(i*k), ((j-1)*k):(j*k)] # k*k     
            vector_ij = block_ij.flatten(order='F') # k^2*1
            R_cal_tmp[(j-1)*p+i-1,:] = vector_ij # the p-th row
    if t < len(R_list) - 1:
        vec_R_cal_tmp = R_cal_tmp.flatten(order="F")
        vec_R_cal_n[t,:] = vec_R_cal_tmp
    else:
        R_cal = R_cal_tmp
        vec_R_cal = R_cal.flatten(order="F")
    count += 1

V = np.cov(vec_R_cal_n, rowvar = False)

L, Sigma_v, Nt = np.linalg.svd(R_cal) ## equation (9)
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