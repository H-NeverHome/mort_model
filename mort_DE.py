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
fin_dat = fin_dat.copy().loc[70::]
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
                  full_matrices=True)
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
forecaster = ARIMA(order=(1, 1, 0), 
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
    
       
#%%
########################################################################
########### MODELING IMPORT
import pickle
import pandas as pd
import os
r_home = r'C:\Program Files\R\R-4.0.2'
os.environ['R_HOME'] = r_home
import rpy2
rpy2_info = (rpy2.__path__,rpy2.__version__,'R_HOME',r_home)
print(rpy2_info)

def r_to_pd(smth_r):
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        smth_pd   = ro.conversion.rpy2py(smth_r)
    return smth_pd

def pd_to_r(smth_pd):
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        smth_r   = ro.conversion.py2rpy(smth_pd)
    return smth_r

from rpy2.robjects import globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

base = importr('base')
utils = importr('utils')
brms = importr('brms')
sjPlot = importr('sjPlot')
ro.r('if (require(StMoMo) == FALSE) {install.packages("StMoMo")}')
ro.r('if (require(demography) == FALSE) {install.packages("demography")}')
ro.r('if (require(HMDHFDplus) == FALSE) {install.packages("HMDHFDplus")}')
importr('demography')
importr('StMoMo')
importr('HMDHFDplus')



# ro.r('''wDE_DAT <- readHMDweb(CNTRY = "DEUTFRG", 
#                               username = "hdgniehaus@arcor.de",
#                               password = "Moi_quiee148")''')

ro.r('''wDE_DAT <- readHMD("Mx_1x1.txt")''')
# ro.r('''aaa <- demogdata(wDE_DAT)''')
ro.r('''wDE <- read.demogdata("Mx_1x1.txt", 
                              "Population.txt", 
                              type="mortality", 
                              label="w_DE")''')

# ro.r('plot(wDE_DAT,series = "total")')
# ro.r('print(names(wDE$rate)[2])')
ro.r('wDEIniData <- StMoMoData(wDE, series = "total",type = "central")')

#%%
ro.r('LC <- lc(link = "logit")')
ro.r('EWMaleIniData <- central2initial(EWMaleData)')
ro.r('EWMaleData$Dxt')
ro.r('EWMaleData$Ext')

ro.r('ages.fit <- 55:89')
ro.r('wxt <- genWeightMat(ages = ages.fit, years = EWMaleIniData$years,clip = 3)')
ro.r('LCfit <- fit(LC, data = EWMaleIniData, ages.fit = ages.fit, wxt = wxt)')
ro.r('plot(LCfit, nCol = 3)')
ro.r('export1 <- EWMaleIniData$Dxt')

ro.r('export2 <- EWMaleIniData$Ext')

aaa = np.array(globalenv['export1'])
bbb = np.array(globalenv['export2'])


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




