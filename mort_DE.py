"""
Created on Sat Aug 27 11:31:37 2022

@author: de_hauk
"""


#%%
########################################
#################### IMPORTS PYTHON
from sktime.forecasting.arima import AutoARIMA,ARIMA
import pandas as pd
import numpy as np
import seaborn as sns
import os 
import rpy2

wdir = r'C:\Users\de_hauk\Documents\GitHub\mort_model'
os.chdir(wdir)
from mort_model_class import LC_model
#%%
########################################
#################### IMPORTS R & RPY2

r_home = r'C:\Program Files\R\R-4.0.2'
os.environ['R_HOME'] = r_home

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
########################################
#################### IMPORT & PREPROCESS DATA FROM HMD DEATHRATES W_GER

data_path = r'Mx_1x1.txt'
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

      



#%%
##############################################################################
############################ GET STANDARD DATA from STMOMO and export to py
#Age-specific deaths and exposures for England and Wales from the Human Mortality Database. 
ro.r('EWMaleIniData <- central2initial(EWMaleData)')

ro.r('exp1 <- EWMaleData$Dxt') #matrix of deaths
ro.r('exp1_row <- rownames(EWMaleData$Dxt)')
ro.r('exp1_clm <- colnames(EWMaleData$Dxt)')

ro.r('exp2 <- EWMaleData$Ext') #matrix of pop
ro.r('exp2_row <- rownames(EWMaleData$Ext)')
ro.r('exp2_clm <- colnames(EWMaleData$Ext)')

n_D = pd.DataFrame(data = r_to_pd(globalenv['exp1']),
                   columns = list(globalenv['exp1_clm']),
                   index = list(globalenv['exp1_row']))
n_POP = pd.DataFrame(data = r_to_pd(globalenv['exp2']),
                     columns = list(globalenv['exp2_clm']),
                     index = list(globalenv['exp2_row']))

d_rate = pd.DataFrame(data = n_D.values/n_POP.values,
                      columns = list(globalenv['exp2_clm']),
                      index = list(globalenv['exp2_row']))

dat_d_rate = d_rate.copy().loc[[str(i) for i in range(55,99)]]



#%%
#Init & FIT R model
ro.r('LC1 <- lc(link = "log")')
ro.r('LCfit <- fit(LC1, data = EWMaleData, ages.fit = 55:99)')
ro.r('plot(LCfit)')
ro.r('LCfit_kt <- LCfit$kt')
ro.r('colnames(LCfit$kt)')
kt_R = np.array(globalenv['LCfit_kt'])

#%%
from mort_model_class import LC_model
#Init Python model
lc_py = LC_model()

lc_py.load_data(data = dat_d_rate,data_type='rate')

# xxxx = lc_py.k_t

# future_lags = [i for i in range(15)]
# arima_model = AutoARIMA(start_p = 0,
#                         random = True,
#                         n_fits = 100,
#                         out_of_sample_size = 5,
#                         information_criterion = 'bic',
#                         method = 'bfgs',
#                         maxiter = 250).fit_predict(xxxx,fh = future_lags)


res_lc_py = lc_py.fit_predict(n_futures = 15, pred_int =.7)

 
lc_py.plot_kt()



#%%

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



# #%%
# plt_dat = pd.DataFrame()
# plt_dat['kt_log'] = lc_res_stmomo.flatten().tolist() + y_pred_arm_R.flatten().tolist()
# plt_dat['kt_upper'] = [0 for i in range(len(lc_res_stmomo.flatten().tolist()))] + high_int
# plt_dat['kt_lower'] = [0 for i in range(len(lc_res_stmomo.flatten().tolist()))] + low_int

# aaaP = sns.lineplot(x       = plt_dat.index.to_list(),
#                     y      = 'kt_log',
#                     data   = plt_dat)

# aaaP.fill_between(x =     plt_dat.index.to_list(), 
#                   y1 =    plt_dat['kt_log']+plt_dat['kt_upper'].to_numpy(), 
#                   y2 =    plt_dat['kt_log']-plt_dat['kt_lower'].to_numpy(),
#                     alpha = .25)




# #%%
# ro.r('kt.LCfit.diff <- apply(kt.LCfit, 2, diff)')
# aaa = np.array(globalenv['kt.LCfit'])
# #ro.r('print(kt.LCfit.diff)')
# #%%
# ro.r('''ar_model <- arima(kt.LCfit, 
#                          method = "ML",
#                          order = c(1, 0, 1))''')

# #%%
# ro.r('''res_ar_model <- predict(ar_model,n.ahead = 15)''')
# bbb = np.array(globalenv['res_ar_model'])

# #%%
# ro.r('pred.ktdiff.LCfit.11 <- VARMApred(fit.kt.LCfit.11, h = h)')
# ro.r('pred.kt.LCfit.11 <- apply(rbind(tail(kt.LCfit, n = 1),')
# ro.r('pred.ktdiff.LCfit')



# #%%


# ro.r('''wDE_DAT <- readHMD("Mx_1x1.txt")''')
# # ro.r('''aaa <- demogdata(wDE_DAT)''')
# ro.r('''wDE <- read.demogdata("Mx_1x1.txt", 
#                               "Population.txt", 
#                               type="mortality", 
#                               label="w_DE")''')

# # ro.r('plot(wDE_DAT,series = "total")')
# # ro.r('print(names(wDE$rate)[2])')
# ro.r('wDEIniData <- StMoMoData(wDE, series = "total",type = "central")')

# #%%


# ro.r('export1 <- EWMaleIniData$Dxt')

# ro.r('export2 <- EWMaleIniData$Ext')

# aaa = np.array(globalenv['export1'])
# bbb = np.array(globalenv['export2'])



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


# # kt_py = lc_py.k_t.copy()
# # kt_py_1 = kt_py.copy()-np.sum(kt_py.copy())
# # kt_py_2 = kt_py_1*np.sum(lc_py.b_x.copy())
# # kt_py_3 = kt_py_2
# # kt_py = lc_py.k_t.copy()
# # kt_py_1 = kt_py-np.sum(kt_py)
# # mata_k = svd_res['U'][:,0]*np.sum(svd_res['V*'][1,:])*svd_res['S'][0]

# #%%
# svd_res = lc_py.res_SVD_fin
# ax = lc_py.a_x
# bx = lc_py.b_x
# s_1 = lc_py.res_SVD_fin['S'][0]
# '''Lee and Carter also take the first column of U, multiply by D1,1 
# and multiply by the sum of the first row of V
# (to cancel the division) and call that k. 
# This vector captures overall mortality change over time.'''

# #. mata k = U[,1] * sum(Vt[1,]) * d[1]

# mata_k = svd_res['U'][:,0]*np.sum(svd_res['V*'][1,:])*svd_res['S'][0]

# mort_data_raw_ = lc_py.mort_data_raw 

# # ro.r('''LCfit <- fit(lc(), Dxt = EWMaleData$Dxt, Ext = EWMaleData$Ext, 
# #              ages = EWMaleData$ages, years = EWMaleData$years, 
# #              ages.fit = 55:89)''')
# # ro.r('plot(LCfit)')





# #%%
# ro.r('ages.fit <- 55:89')
# ro.r('wxt <- genWeightMat(ages = ages.fit, years = EWMaleIniData$years,clip = 3)')
# # aaa =  r_to_pd(globalenv['EWMaleIniData'])
# ro.r('LC <- lc(link = "logit")')
# ro.r('LCfit <- fit(LC, data = EWMaleIniData, ages.fit = ages.fit, wxt = wxt)')
# ro.r('plot(LCfit, nCol = 3)')
# ro.r('exxp_1 <- LCfit$Dxt')
# aaa =  r_to_pd(globalenv['EWMaleIniData'])
# ro.r('h <- 15')
# ro.r('kt.LCfit <- t(LCfit$kt)')


# #%%
# lc_inst = LC_model()
# lc_inst.load_data(fin_dat, data_type='rate')
# data_raw = lc_inst.mort_data_raw

# #extract param k_t
# xx_kt = lc_inst.k_t
# #extract param b_x
# xx_bx = lc_inst.b_x
   
# zzz_pred = lc_inst.fit_predict(n_futures = 15, pred_int =.7) 
# lc_inst.plot_kt()

  