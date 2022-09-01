

class LC_model:
    
    def __init__(self):
        print('model init')
 
        self.fit = False
    
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
            self.k_t = self.res_SVD_fin['U'][:, 0]-np.mean(self.res_SVD_fin['U'][:, 0])

            #extract param b_x
            self.b_x = pd.DataFrame(self.res_SVD_fin['V*']).T[0].values        
        else:
            print()
        
    def fit_predict(self,n_futures, pred_int):
        import pandas as pd
        import numpy as np
        from scipy.linalg import svd
        from sktime.forecasting.arima import ARIMA,AutoARIMA
        self.future_N = n_futures
        self.pred_interval = pred_int
        arima_model = AutoARIMA(start_p = 0,
                                random = True,
                                n_fits = 100,
                                out_of_sample_size = 5,
                                information_criterion = 'bic',
                                method = 'bfgs',
                                maxiter = 250).fit(self.k_t)
        
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
        self.fit = True
        return self.arima_results

    def plot_kt(self):
        import pandas as pd
        import numpy as np
        from scipy.linalg import svd
        import seaborn as sns
        if self.fit == False:
            print('ERROR ARIMA NOT FIT')
        elif self.fit == True:
	    sns.set(rc={'figure.figsize':(20,15)})
            plt_ = self.arima_results['y_pred_intervals']
            plt_sns = sns.lineplot(x      = plt_.index.to_list(),
                                   y      = 'kt_log',
                                   data   = plt_)
            
            plt_sns.fill_between(x =     plt_.index.to_list(), 
                                 y1 =    plt_['kt_log']+plt_[f'higher_{self.pred_interval}'].to_numpy(), 
                                 y2 =    plt_['kt_log']-plt_[f'lower_{self.pred_interval}'].to_numpy(),
                                 alpha = .25)
            plt_sns.tick_params(axis='x', rotation=45)
            plt_sns.invert_yaxis()
            #plt_sns.set_xticklabels(plt_sns.get_xticklabels(),rotation=30) 
  


# def lee_carter(rate, T, N, misc=False):

#     logm_xt = np.log(rate).T

#     a_x = logm_xt.sum(axis=1) / T
#     z_xt = logm_xt - a_x.reshape(N, 1)

#     U, S, V = np.linalg.svd(z_xt, full_matrices=True)

#     bxkt = S[0] * np.dot(U[:, 0].reshape(N, 1), V[0, :].reshape(T, 1).T)
#     eps = z_xt - bxkt

#     logm_xt_lcfitted = bxkt + a_x.reshape(N, 1)

#     b_x = U[:, 0]/U[:, 0].sum()
#     k_t = V[0, :]*S[0]*U[:, 0].sum()
#     a_x = a_x + k_t.sum()*b_x
#     #k_t = k_t - k_t.sum()

#     kwargs = {"U": U, "S": S, "V": V, "logm_xt": logm_xt,
#               "z_xt": z_xt, "eps": eps, "logm_xt_lcfitted": logm_xt_lcfitted}

#     return (a_x, b_x, k_t, kwargs) if misc else (a_x, b_x, k_t)

# aaaa =  lee_carter(rate = dat_d_rate.values, 
#                    T = len(dat_d_rate.index.tolist()), 
#                    N = len(dat_d_rate.columns.tolist()), 
#                    misc=True)


