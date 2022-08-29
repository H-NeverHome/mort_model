# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:31:37 2022

@author: de_hauk
"""
from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA,ARIMA
import pandas as pd
import numpy as np
import seaborn as sns

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


class LC_model:
    
    def __init__(self):
        print('model init')
        import pandas as pd
        import numpy as np
        from scipy.linalg import svd

    
    def load_data(self,data,data_type):
        import pandas as pd
        import numpy as np
        from scipy.linalg import svd
        error_msg_datatype = 'input/data-type not supported.'
        error_mdg_data = '''Data needs to be dataframe of mortality rates.
                            Years as COLUMNS and Age as ROWS'''
        if data_type == 'rate':
            self.mort_data_raw = data.dropna(axis = 0,
                                             how = 'any').apply(func = np.log,
                                                                axis = 1,
                                                                raw = True)
                                                  
            self.a_x = self.mort_data_raw.mean(axis = 'columns')
            
            a_xt_raw = pd.DataFrame(index = [i for i in self.mort_data_raw.index])
            for year in self.mort_data_raw:
                curr_clm = self.mort_data_raw[year].values - self.a_x.values
                a_xt_raw[year] = curr_clm.tolist()
            
            
            self.a_xt = a_xt_raw.copy().T
            
            res_SVD = svd(self.a_xt.values,
                          full_matrices=True)     
            
            ### Compute SVD of A_xt
            # USV* = SVD(A_xt)  
            # U == left-singular vectors
            # S == singular values 
            # V* == right-singular vectors                                               
            self.res_SVD_fin = {'U' : res_SVD[0],
                                'S' : res_SVD[1],
                                'V*' : res_SVD[2]}
            
            #extract param k_t
            self.k_t = pd.DataFrame(self.res_SVD_fin['U'])[0].values
            
            #extract param b_x
            self.b_x = pd.DataFrame(self.res_SVD_fin['V*']).T[0].values        
        else:
            print()
        
    def fit_predict(self,n_futures, pred_int):
        from sktime.forecasting.arima import ARIMA
        self.future_N = n_futures
        
        arima_model = ARIMA(order=(1, 0, 1), 
                            seasonal_order=(0, 0, 0, 0)).fit(self.k_t)
        
        self.arima_model = arima_model
        self.future_lags = [i for i in range(self.future_N)]
        self.y_pred = arima_model.predict(fh=self.future_lags).flatten().tolist()
        self.y_pred_int = arima_model.predict_interval(fh = self.future_lags,
                                                          coverage = pred_int)

        
        self.mort_data = self.mort_data_raw.copy()
        last_dat = int(self.mort_data.columns.max())
        
        for fut_year in range(self.future_N):
            term1 = self.a_x+self.res_SVD_fin['S'][0]
            term2 = self.b_x*self.y_pred[fut_year]
            term3 = term1*term2
            self.mort_data[str(last_dat+(fut_year+1))] = term3
       
        self.all_Y = self.mort_data.columns.tolist()
        final_res_kt = pd.DataFrame(index = self.all_Y)
        final_res_kt['kt_log'] = self.k_t.flatten().tolist() + self.y_pred
        print(final_res_kt)
        for i,j in zip(self.y_pred_int,[f'lower_{pred_int}',f'higher_{pred_int}']):
            dat = self.y_pred_int[i].tolist() + self.y_pred
            final_res_kt[j] = [0 for i in range(len(self.k_t.flatten().tolist()) )]+self.y_pred_int[i].tolist()
       

        self.arima_results = {'y_pred'          : self.y_pred,
                             'y_pred_intervals' : final_res_kt,
                             'params'           : arima_model.get_fitted_params(),
                             'pred_mort_rates'  : np.exp(self.mort_data)}       
        return self.arima_results


        
#%%


lc_inst = LC_model()
lc_inst.load_data(fin_dat, data_type='rate')
data_raw = lc_inst.mort_data_raw

#extract param k_t
xx_kt = lc_inst.k_t
#extract param b_x
xx_bx = lc_inst.b_x
   
zzz_pred = lc_inst.fit_predict(n_futures = 15, pred_int =.9) 

#%% 
plt_dat = zzz_pred['y_pred_intervals']
aaaP = sns.lineplot(x      = plt_dat.index.to_list(),
                    y      = 'kt_log',
                    data   = plt_dat)

aaaP.fill_between(x =     plt_dat.index.to_list(), 
                  y1 =    plt_dat['kt_log']+plt_dat['higher_0.9'].to_numpy(), 
                  y2 =    plt_dat['kt_log']-plt_dat['lower_0.9'].to_numpy(),
                  alpha = .25)





#%% 
######################################################################
################# USE ARIMA TO PREDICTI
######################################################################
####predict

y = load_airline()
 
 

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
ro.r('if (require(MTS) == FALSE) {install.packages("MTS")}')
ro.r('if (require(stats) == FALSE) {install.packages("stats")}')

importr('MTS')
importr('demography')
importr('StMoMo')
importr('HMDHFDplus')
importr('stats')
#%%
##############################################################################
############################ Fit Standard data from STMOMO
ro.r('LC <- lc(link = "logit")')
ro.r('EWMaleIniData <- central2initial(EWMaleData)')

ro.r('EWMaleData$Dxt')
ro.r('EWMaleData$Ext')

ro.r('ages.fit <- 55:89')
ro.r('wxt <- genWeightMat(ages = ages.fit, years = EWMaleIniData$years,clip = 3)')
ro.r('LCfit <- fit(LC, data = EWMaleIniData, ages.fit = ages.fit, wxt = wxt)')
ro.r('plot(LCfit, nCol = 3)')

ro.r('h <- 15')
ro.r('kt.LCfit <- t(LCfit$kt)')

lc_res_stmomo = np.array(globalenv['kt.LCfit'])

n_future_years_R = 15

arm_R = ARIMA(order=(1, 0, 1), 
              seasonal_order=(0, 0, 0, 0))

arm_R.fit(lc_res_stmomo)  

future_H = [i for i in range(n_future_years_R)]

y_pred_arm_R = arm_R.predict(fh=future_H) 
y_pred_L_arm_R = list(y_pred)

suppl_param = arm_R.get_fitted_params() 
p_interval_R = arm_R.predict_interval(fh = future_H)  

indx_predint = list(p_interval_R.columns)
high_int = p_interval_R[indx_predint[0]].tolist()
low_int = p_interval_R[indx_predint[1]].tolist()

#%%
plt_dat = pd.DataFrame()
plt_dat['kt_log'] = lc_res_stmomo.flatten().tolist() + y_pred_arm_R.flatten().tolist()
plt_dat['kt_upper'] = [0 for i in range(len(lc_res_stmomo.flatten().tolist()))] + high_int
plt_dat['kt_lower'] = [0 for i in range(len(lc_res_stmomo.flatten().tolist()))] + low_int

aaaP = sns.lineplot(x       = plt_dat.index.to_list(),
                    y      = 'kt_log',
                    data   = plt_dat)

aaaP.fill_between(x =     plt_dat.index.to_list(), 
                  y1 =    plt_dat['kt_log']+plt_dat['kt_upper'].to_numpy(), 
                  y2 =    plt_dat['kt_log']-plt_dat['kt_lower'].to_numpy(),
                    alpha = .25)




#%%
ro.r('kt.LCfit.diff <- apply(kt.LCfit, 2, diff)')
aaa = np.array(globalenv['kt.LCfit'])
#ro.r('print(kt.LCfit.diff)')
#%%
ro.r('''ar_model <- arima(kt.LCfit, 
                         method = "ML",
                         order = c(1, 0, 1))''')

#%%
ro.r('''res_ar_model <- predict(ar_model,n.ahead = 15)''')
bbb = np.array(globalenv['res_ar_model'])

#%%
ro.r('pred.ktdiff.LCfit.11 <- VARMApred(fit.kt.LCfit.11, h = h)')
ro.r('pred.kt.LCfit.11 <- apply(rbind(tail(kt.LCfit, n = 1),')
ro.r('pred.ktdiff.LCfit')



#%%
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


ro.r('export1 <- EWMaleIniData$Dxt')

ro.r('export2 <- EWMaleIniData$Ext')

aaa = np.array(globalenv['export1'])
bbb = np.array(globalenv['export2'])



            # ########## mu3
            # mu3_plot_DF = pd.DataFrame()
            # mu3_plot_DF['mu3'] = post_hid_st['mu3'].flatten()
            # mu3_plot_DF['sa3'] = post_hid_st['sa3'].flatten()
            
            # sns.lineplot(x      = mu3_plot_DF.index.to_list(),
            #               y      = 'mu3',
            #               data   = mu3_plot_DF,
            #               ax     = axs[0])
            
            # axs[0].fill_between(x =     mu3_plot_DF.index.to_list(), 
            #                     y1 =    mu3_plot_DF['mu3']+mu3_plot_DF['sa3'].to_numpy(), 
            #                     y2 =    mu3_plot_DF['mu3']-mu3_plot_DF['sa3'].to_numpy(),
            #                     alpha = .25)
            # axs[0].set(xlabel='trials', 
            #             ylabel='mu3+-sd3')

# #### Only use full colums
# fin_dat = fin_dat.copy().dropna(axis = 0,
#                                  how = 'any')
# fin_dat = fin_dat.copy().loc[70::]
# ### transform to log
# fin_dat_log = fin_dat.copy().apply(func = np.log,
#                                    axis = 1,
#                                    raw = True)


# n_future_years = 15
# forecaster = ARIMA(order=(1, 0, 1), 
#                    seasonal_order=(0, 0, 0, 0))
# forecaster.fit(k_t)  
# future_H = [i for i in range(n_future_years)]
# y_pred = forecaster.predict(fh=future_H) 
# y_pred_L = list(y_pred)

# fin_data_pred = fin_dat.copy()
# last_dat = int(fin_dat.columns.max())

# for fut_year in range(n_future_years):
#     term1 = a_time+res_SVD_scp_fin['S'][0]
#     term2 = b_x*y_pred_L[fut_year]
#     term3 = np.exp(term1*term2)
#     fin_data_pred[str(last_dat+(fut_year+1)+1)] = term3
    
# zzzz = forecaster.get_fitted_params() 
# xxxx = forecaster.predict_interval(fh = future_H)   

# #extract param k_t
# k_t = pd.DataFrame(res_SVD_scp_fin['U'])[0].values
# xx_kt = lc_inst.k_t
# #extract param b_x
# b_x = pd.DataFrame(res_SVD_scp_fin['V*']).T[0].values
# xx_bx = lc_inst.b_x
 
  