# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:31:37 2022

@author: de_hauk
"""

import pandas as pd
import numpy as np

data_path = r'C:\Users\de_hauk\Documents\GitHub\mort_model/Mx_1x1.txt'
raw_dat_wDE = pd.read_csv(filepath_or_buffer = data_path,
                            sep= '\t',
                            header = 1,
                            engine = 'python',
                            encoding = 'utf_8',
                            dtype = str)

proc_dat = pd.DataFrame()

res = []
for row_nr,row_cont in raw_dat_wDE.iterrows():
    i_proc = [i for i in list(row_cont)[0].split(' ') if len(i)>=1]
    res.append(i_proc)
    proc_dat[str(row_nr)] = i_proc
    
proc_dat_T = pd.DataFrame(proc_dat.T.values,
                          columns = ['Y','AGE','F','M','TOTAL'])
 

proc_dat_T = proc_dat_T.astype('str',
                               copy=True)

proc_dat_T.replace(['.'], [np.nan],
                   inplace = True) 


proc_dat_T['F'] = proc_dat_T['F'].astype(float, 
                        errors = 'ignore',
                        copy=False)

proc_dat_T['M'] = proc_dat_T['M'].astype(float, 
                        errors = 'ignore',
                        copy=False)

proc_dat_T['TOTAL'] = proc_dat_T['TOTAL'].astype(float, 
                            errors = 'ignore',
                            copy=False)

proc_dat_T.drop(['F','M'],
                axis = 1,
                inplace = True)


fin_dat = proc_dat_T.pivot(index='Y', columns='AGE')['TOTAL'].T

new_indx = [int(i[0:3]) for i in fin_dat.index]
fin_dat['new_indx'] = new_indx
fin_dat.set_index('new_indx',inplace = True)

fin_dat.sort_index(axis=0,
                   ascending=True,
                   inplace=True)
fin_dat[fin_dat>1.] = np.nan
fin_dat[fin_dat==0] = np.nan


# import seaborn as sns
# sns.set_context('paper')
# ax = sns.heatmap(fin_dat, 
#                  linewidths=.5,
#                  annot=False,
#                  xticklabels = [i for i in fin_dat],
#                  yticklabels = [i for i in fin_dat.index],
#                  square = True)
#%%
# ##### Fillna
# for i in fin_dat:
#     fin_dat[i] = fin_dat[i].copy().fillna(fin_dat[i].max())

#### Only use full colums
fin_dat = fin_dat.copy().dropna(axis = 0,
                                 how = 'any')

### transform to log
fin_dat_log = fin_dat.copy().apply(func = np.log,
                                   axis = 1,
                                   raw = True)

### compute a_x
a_time = fin_dat_log.copy().mean(axis = 'columns')

### compute A_xt
A_xt_raw = pd.DataFrame(index = [i for i in fin_dat.index])
for i in fin_dat:
    curr_clm = fin_dat_log[i].values - a_time.values
    A_xt_raw[i] = curr_clm.tolist()
    
    
### Compute SVD of A_xt
# USV* = SVD(A_xt)  
# U == left-singular vectors
# S == singular values 
# V* == right-singular vectors

A_xt = A_xt_raw.copy().T

from scipy.linalg import svd
res_SVD_scp = svd(A_xt.values,
                  full_matrices=False)
res_SVD_scp_fin = {'U' : res_SVD_scp[0],
                   'S' : res_SVD_scp[1],
                   'V*' : res_SVD_scp[2]}

#extract param k_t
k_t = pd.DataFrame(res_SVD_scp_fin['U'])[0].values

#extract param b_x
b_x = pd.DataFrame(res_SVD_scp_fin['V*']).T[0].values

   

#%% 
######################################################################
################# USE ARIMA TO PREDICTI
######################################################################
####predict
from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA,ARIMA
y = load_airline()

n_future_years = 5
forecaster = ARIMA(order=(1, 1, 1), 
                   seasonal_order=(0, 0, 0, 0))
forecaster.fit(k_t)  
y_pred = forecaster.predict(fh=[i for i in range(n_future_years)]) 
y_pred_L = list(y_pred)

fin_data_pred = fin_dat.copy()
last_dat = int(fin_dat.columns.max())

for fut_year in range(n_future_years):
    term1 = a_time+res_SVD_scp_fin['S'][0]
    term2 = b_x*y_pred_L[fut_year]
    term3 = np.exp(term1*term2)
    fin_data_pred[str(last_dat+(fut_year+1)+1)] = term3
    
       

    
    
    # age_mr = []
    # for age in fin_dat.index:
    #     term1 = a_time.loc[age]+res_SVD_scp_fin['S'][0]
    #     term2 = y_pred_L[fut_year]*b_x[age]
    #     term3 = np.exp(term1*term2)
    #     age_mr.append(float(term3))
  
    # fin_data_pred[str(last_dat+age)] = age_mr
    
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=len([i for i in fin_dat.index]), n_iter=50, random_state=42)   
# res_svd = svd.fit_transform(A_xt_raw.T.values)
# params_SVD = svd.get_params()

# a_xt = fin_dat_log.subtract(a_time,
#                    axis = 1)
#astype(int)
# fin_dat = pd.melt(proc_dat_T, id_vars=['Y','AGE'], value_vars=['TOTAL'])




