import pandas as pd

import numpy as np
# random seed for numpy --> random seed for sklearn, pandas, statsmodels
np.random.seed(99)

import random
random.seed(99)

import matplotlib.pyplot as plt


# Feature Transformations
# from sktime.transformations.series.feature_selection import FeatureSelection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# MICE-Imputation
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

# Statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX # statespace alternative
# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

def abs_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def squared_error(y_true, y_pred):
    return (y_true - y_pred)**2

def tadda_error(y_true: np.array, y_pred: np.array):
    """
    Targeted Absolute Distance with Direction Augmentation (TADDA)
    """
    threshold = 0.048
    
    abs_eror_part = abs_error(y_true, y_pred)
        
    wrong_signs = (np.sign(y_pred) * np.sign(y_true) == -1).astype('int') # 0 if correct sign; 1 if wrong sign 
    significant_diff = (abs_eror_part > threshold).astype('int')
    
    # mae in case wrong sign and over threshold
    sign_loss = (np.abs(y_pred) * wrong_signs * significant_diff)
    
    # print("---------------------------------------------------")
    # print(mae_part)
    # print("############")
    # # print(np.sign(y_pred))
    # # print(np.sign(y_true))
    # # print(wrong_signs)
    # # print("############")
    # print(significant_diff)
    # print("############")
    # print(sign_loss)
    # print("---------------------------------------------------")

    return abs_eror_part + sign_loss

def tsdda_error(y_true: np.array, y_pred: np.array):
    """
    Targeted Squared Distance with Direction Augmentation (TSDDA)
    """
    threshold = 0.048
    
    squared_eror_part = squared_error(y_true, y_pred)
        
    wrong_signs = (np.sign(y_pred) * np.sign(y_true) == -1).astype('int') # 0 if correct sign; 1 if wrong sign 
    significant_diff = (squared_eror_part > threshold).astype('int')
    
    # mae in case wrong sign and over threshold
    sign_loss = (np.abs(y_pred) * wrong_signs * significant_diff)
    
    # print("---------------------------------------------------")
    # print(mae_part)
    # print("############")
    # # print(np.sign(y_pred))
    # # print(np.sign(y_true))
    # # print(wrong_signs)
    # # print("############")
    # print(significant_diff)
    # print("############")
    # print(sign_loss)
    # print("---------------------------------------------------")

    return squared_eror_part + sign_loss








from abc import ABC, abstractmethod
class ParametricTimeSeriesForecaster(ABC):
    
    @abstractmethod
    def fit(self, X_train, y_train, params):
        pass
    
    @abstractmethod
    def predict(self, X_test, fh, ci = False):
        pass
    
    @abstractmethod
    def predict_insample(self):
        pass
    
    
    

########################################
#           NO-CHANGE MODEL            # 
########################################

class NoChange():
    
    def __init__(self):
        
        self.y_t0 = None
        self.y_train = None

        self.is_fitted = False

    def fit(self, X_train, y_train):
        
        self.y_t0 = y_train.iloc[-1, 0]
        self.y_train = y_train
        self.is_fitted = True
        
        return self
        
#     def predict(self, X_test, fh, ci = False):
        
#         y_hat0 = np.empty(fh)
#         y_hat0[:] = self.y_t0 # last observed value carried forward
        
#         if ci == True:
#             return y_hat0, None
#         else:
#             return y_hat0

    def predict(self, X_test, fh):

        y_hat0 = np.empty(fh)
        y_hat0[:] = self.y_t0 # last observed value carried forward

        return y_hat0
    
#     def predict_insample(self):
        
#         y_hat = np.empty(self.y_train.shape[0])
#         y_hat[:] = self.y_t0 # last observed value carried forward
        
#         y_hat = pd.DataFrame(y_hat, index = self.y_train.index)
        
#         return y_hatDie
    
    def evaluate_model(self, X_test, y_test, fh):
        
        if self.is_fitted is True: # check if model is trained
            
            ### 1-YEAR-AHEAD PREDICITON
            # Based on the nature of the ARIMA equations, out-of-sample forecasts tend to converge to the sample mean for long forecasting periods.
            # print("Based on the nature of the ARIMA equations, out-of-sample forecasts tend to converge to the sample mean for long forecasting periods.")
        
            forecast = self.predict(X_test, fh)
            forecast = pd.DataFrame(forecast, index = X_test.index)
            
            ### LOG-CHANGE CALCULATION
            y_test_logchange = y_test - self.y_t0
            forecast_logchange = forecast - self.y_t0
            
            # print(forecast)
            # print(y_test)

            
            ### PLOT TRAININGSDATA AND 
            fig, ax = plt.subplots(figsize=(15, 10))
            
            self.y_train.plot(ax=ax, label="y_train", marker="o", color="blue")
            y_test.plot(ax=ax, label="y_test", marker="o",color="lightblue")

            ### PLOT OUT-OF-SAMPLE PREDICTION RESULTS
            # plt.plot(y_test.index, forecast, label="predicted mean", marker="o", color="red")
            forecast.plot(label="predicted mean", marker="o", color="red", ax = ax)


            # ax.fill_between(conf_int.index, conf_int["mean_ci_lower"], conf_int["mean_ci_upper"], alpha=0.1, label="90%-confidence intervall")
            
            plt.title("In-sample fit and out-of-sample prediction (log. abs. values): " + str(y_test.columns[0]))
            plt.grid()
            plt.legend([y_test.columns[0], y_test.columns[0], "predicted mean"])
            plt.show()
            
            ### LOG-CHANGE FORECAST VISUALIZATION
            fig, ax = plt.subplots(figsize=(15, 10))
            y_test_logchange.plot(ax=ax, marker="o", color="lightblue") 
            forecast_logchange.plot(marker="o", color="red", ax = ax) 

            plt.title("Out-of-sample predictions (log-change): " + str(y_test_logchange.columns[0]))
            plt.legend(["y_test", "y_pred"])
            plt.grid()
            plt.show()
            
            ### BUILD PREDICITON RESULT DF FOR MODEL EVALUATION
            table = pd.DataFrame()
            table["MONTH"] = y_test.index
            table["FAT_PRED"] = forecast.to_numpy()
            table["FAT_ACTUAL"] = y_test.to_numpy()
            # log change
            table["LC(FAT_PRED)"] = forecast_logchange.to_numpy()
            table["LC(FAT_ACTUAL)"] = y_test_logchange.to_numpy()
 
            # print(table)
            ### OUT-OF-SAMPLE MODEL EVALUATION VIA TADDA-SCORE
            # print(table)
            tadda = np.mean(table.apply(lambda x: tadda_error(x["LC(FAT_ACTUAL)"], x["LC(FAT_PRED)"]), axis=1))
            print("TADDA: "+ str(tadda))
            print("Evaluation finished.")
            return table
        
        else:
            print("Model needs to be fitted before it can be evaluated.")
            print("Evaluation finished.")

            return None
    
    
###########################################################################
#           TimeSeriesForecasterPCA ENSEMBLE MODEL with MICE IMPUTATION   #
###########################################################################


# class TimeSeriesForecasterPCA(ParametricTimeSeriesForecaster) :
        
#     def __init__(self):
            
#         self.fitted_model = None
        
#         self.MICEImputer = None
#         self.scaler = None
#         self.pca = None
        
#         self.X_train_preprocessed = None   
#         self.y_train = None
    
#     def fit(self, X_train, y_train, params):
#         """
#         1. check if X and y are in the right format --> reformat
#         2. Model Decision based on y obs availability: ARIMA or No Change
#         _______________________
#             CASE 1: ARIMA
#                 2.1 Imputation of X:
#                     2.1.1 remove columns with too many NaNs
#                     2.1.2 linear interpolation
#                 2.2 Clean Data: remove constant values, ...
#                 2.3 Align Data indices: best starting date heuristic with minimal data column loss
#                 2.4 Scaling
#                 2.5 PCA
#                 2.6 ARIMA fit
#         _______________________
#             CASE 2: NoChange
#                 2.2 NoChange FIT
#        _______________________
#             CASE 3: No Model
#                 2.3 Return None
                
#         """
        
#         # print("MODEL FIT")
#         # print(X_train)
#         # print(y_train)
#         # print("--------------------------------------------------")
        
#         self.y_train = y_train
        
#         ### UNPACKING PARAMETERS
#         n_components = params[0]
#         order = params[1]
#         seasonal_order = params[2]
#         trend = params[3]
        
#         ###################################### TODO:
#         ### IF insufficient training data --> NoChange
#         ########################################################
        
#         ### X CHECKING AND REFORMATTING
#         if X_train is not None and X_train.empty: 
#             # no X features available for this country -> fit model without X
#             X_train = None
        
#         ############### PIPELINE ###############
        
#         ### MICE-IMPUTATION with LinearRegression with 5 features
#         if X_train is not None:
#             self.MICEImputer = IterativeImputer(estimator=LinearRegression(),
#                                                 n_nearest_features=5, 
#                                                 random_state=0,
#                                                 verbose=0)
#             print("MICE FIT TRANSFORM")
#             X_train_preprocessed = self.MICEImputer.fit_transform(X_train)
#             if X_train_preprocessed.shape[1] == 0:
#                 X_train_preprocessed = None
#         else:
#             X_train_preprocessed = None
        
#         ### CLEAN DATA: Filter out constant columns that cause problems in ARIMA
#         # constant_columns = X_train_preprocessed.columns[np.var(X_train_preprocessed) == 0] # drop columns with var == 0
#         # X_train_preprocessed = X_train_preprocessed.drop(constant_columns, axis=1)
        
#         ### TRANSFORMATION OF X: StandardScaler
#         if X_train_preprocessed is not None:
#             self.scaler = StandardScaler()
#             X_train_preprocessed = self.scaler.fit_transform(X_train_preprocessed)

#         ### DIMENSIONALITY REDUCTION OF X: PCA
#         if n_components != 0 and X_train_preprocessed is not None and X_train_preprocessed.shape[1]>=n_components:
#             self.pca = PCA(n_components=n_components)
#             X_train_preprocessed = self.pca.fit_transform(X_train_preprocessed)
#         else:
#             X_train_preprocessed = None

#         self.X_train_preprocessed = X_train_preprocessed
        
#         # if y_train is not None:
#         #     print("y Data from: "+str(y_train.index[0]))
#         #     print("y Data until: "+str(y_train.index[-1]))
        
#         ### ARIMA
#         if y_train is None:
#             self.fitted_model = None
#             print("y_train is None --> No ARIMA model could be fitted. --> model=None")
#         else:
#             try: 
#                 self.fitted_model = SARIMAX(endog = y_train,
#                                           exog = X_train_preprocessed,
#                                           order = order, 
#                                           seasonal_order = seasonal_order,
#                                           trend = trend
#                                          ).fit()
#             except:
#                 self.fitted_model = None


#         #########################################
            
        
#         return self.fitted_model
    
    
    
#     def predict(self, X_test, fh, ci = False):
        
#         ############### PIPELINE ###############
    
#         ### X CHECKING AND REFORMATTING
#         if X_test is not None and X_test.empty: 
#             # no X features available for this country
#             X_test = None
            
# #         ### IMPUTATION & FEATURE SYNCHRONIZATION
# #         if self.X_train_preprocessed is None:
# #             X_test_preprocessed = None
# #         else:
# #             print(X_test.shape)
# #             X_test_preprocessed = X_test[self.X_train_selected_cols]
# #             print(X_test.shape)

# #         # linear imputation
# #         if X_test_preprocessed is not None:
# #             X_test_preprocessed = X_test_preprocessed.interpolate(method="linear", axis=0, limit_direction="both")

#         ### MICE-IMPUTATION
#         if X_test is not None and self.MICEImputer is not None:
#             print("MICE TRANSFORM")
#             X_test_preprocessed = self.MICEImputer.transform(X_test)
#         else:
#             X_test_preprocessed = None
        
#         ### TRANSFORMATION: StandardScaler
#         if X_test_preprocessed is not None and self.scaler is not None:
#             X_test_preprocessed = self.scaler.transform(X_test_preprocessed)
        
#         ### PCA
#         if self.pca is not None and X_test is not None:
#             X_test_preprocessed = self.pca.transform(X_test_preprocessed)
#         else:
#             X_test_preprocessed = None # if n_components==0
        
#         ### ARIMA
#         forecast = self.fitted_model.get_forecast(steps=fh, exog=X_test_preprocessed)
#         y_pred = forecast.predicted_mean
#         # set negative predictions to zero
#         y_pred = y_pred.clip(lower=0)

#         if ci == True:
#             conf_int = forecast.summary_frame(alpha=0.1)
#             return y_pred, conf_int
        
#         else:
#             return y_pred
    
#     def predict_insample(self):
        
#         forecast = self.fitted_model.predict(steps=self.y_train.shape[0], exog = self.X_train_preprocessed)
#         y_hat = forecast
        
#         return y_hat

    
    
    
################################################################################
#           TimeSeriesForecasterPCA ENSEMBLE MODEL with Linear Interpolation   #
################################################################################


class TimeSeriesForecasterPCA_LinInt(ParametricTimeSeriesForecaster) :
        
    def __init__(self):
            
        self.fitted_model = None
        
        self.scaler = None
        self.pca = None
        
        self.X_train = None
        self.X_train_imputed = None
        self.X_train_preprocessed = None  
        self.X_train_selected_cols = None
        
        self.y_train = None
    
    def fit(self, X_train, y_train, params):
        
#         print("MODEL FIT")
#         print(X_train.shape)
#         print(y_train.shape)
        
        self.y_train = y_train
        
        ### UNPACKING PARAMETERS
        n_components = params[0]
        order = params[1]
        seasonal_order = params[2]
        trend = params[3]
        
        ### X CHECKING AND REFORMATTING
        if X_train is not None and X_train.empty: 
            # no X features available for this country -> fit model without X
            X_train_preprocessed = None
        else:
            X_train_preprocessed = X_train
            
        # print(X_train_preprocessed)

        
        ############### PIPELINE ###############
                
        ### LINEAR INTERPOLATION
        
        # delete all full NaN columns
        if X_train_preprocessed is not None:
            X_train_preprocessed = X_train.dropna(axis=1, how='all')
            self.X_train_selected_cols = X_train_preprocessed.columns
        
        # linear imputation
        if X_train_preprocessed is not None:
            # print("INTERPOL")
            X_train_preprocessed = X_train_preprocessed.interpolate(method="linear", axis=0, limit_direction="both")
            
            if X_train_preprocessed.shape[1] == 0:
                X_train_preprocessed = None
            
        self.X_train_imputed = X_train_preprocessed
        
                
#         if X_train_preprocessed is not None:
#             pd.DataFrame(X_train_preprocessed).plot(figsize=(15,15))
             
#         ## CLEAN DATA: Filter out constant columns that cause problems in ARIMA
#         constant_columns = X_train_preprocessed.columns[np.var(X_train_preprocessed) == 0] # drop columns with var == 0
#         X_train_preprocessed = X_train_preprocessed.drop(constant_columns, axis=1)
        
#         ## TRANSFORMATION OF X: Percentage Change
           
#         print("X INTERPOLATED")
#         print(X_train_preprocessed)
        
#         imf_wb_mask = list(np.logical_or(X_train_preprocessed.columns.str.startswith("WB_"), X_train_preprocessed.columns.str.startswith("IMF_")))
        
#         imf_wb_mask = X_train_preprocessed.columns.str.startswith("WB_") | X_train_preprocessed.columns.str.startswith("IMF_")

# #         print(imf_wb_mask)
#         periods = seasonal_order[3]
# #         print(periods)
        
#         X_train_preprocessed.loc[:, imf_wb_mask] = X_train_preprocessed.loc[:, imf_wb_mask].diff(periods=periods) / periods
#         X_train_preprocessed.loc[:, ~imf_wb_mask] = X_train_preprocessed.loc[:, ~imf_wb_mask].diff(periods=1) / 1.0

#         # remove first 12 nan rows
#         print(X_train_preprocessed.shape)
#         print(y_train.shape)
#         X_train_preprocessed = X_train_preprocessed.iloc[12:, :]
#         y_train = y_train.iloc[12:, :]
#         self.y_train = y_train
        
        
#         print(X_train_preprocessed.shape)
#         print(y_train.shape)
        
# #         print(X_train_preprocessed)
        
                
        
        
        
        
        
        ### TRANSFORMATION OF X: StandardScaler
        if X_train_preprocessed is not None:
            self.scaler = StandardScaler()
            X_train_preprocessed = self.scaler.fit_transform(X_train_preprocessed)

        ### DIMENSIONALITY REDUCTION OF X: PCA
        if n_components != 0 and X_train_preprocessed is not None and X_train_preprocessed.shape[1]>=n_components:
            self.pca = PCA(n_components=n_components)
            X_train_preprocessed = self.pca.fit_transform(X_train_preprocessed)
        else:
            X_train_preprocessed = None

        self.X_train_preprocessed = X_train_preprocessed
        
#         if X_train_preprocessed is not None:
#             pd.DataFrame(X_train_preprocessed).plot(figsize=(15,15))
        
#         if y_train is not None:
#             print("y Data from: "+str(y_train.index[0]))
#             print("y Data until: "+str(y_train.index[-1]))
#             print("X Data from: "+str(y_train.shape))

#         if X_train_preprocessed is not None:
#             print("X Data from: "+str(X_train_preprocessed.shape))
                
        ### ARIMA
        if y_train is None:
            self.fitted_model = None
            print("y_train is None --> No ARIMA model could be fitted. --> model=None")
        else:
            try: 
                self.fitted_model = SARIMAX(endog = y_train,
                                          exog = X_train_preprocessed,
                                          order = order, 
                                          seasonal_order = seasonal_order,
                                          trend = trend
                                         ).fit(disp=0)
            except:
                self.fitted_model = None


        #########################################
#         print("--------------------------------------------------------")
        return self.fitted_model
    
    
    
    def predict(self, X_test, fh, ci = False):
        
        print("PREDICTION#################################")
        print(X_test)
        
        ############### PIPELINE ###############
    
        ### X CHECKING AND REFORMATTING
        if self.X_train_preprocessed is None:
            X_test_preprocessed = None
            # print("X train is none --> X_test is None")
            
        elif X_test is not None and X_test.empty: 
            # no X features available for this country
            X_test_preprocessed = None
        else:
            ### DATA LEAKAGE PREVENTION ###################################################################
            X_test_preprocessed = X_test
            X_test_preprocessed.iloc[:, :] = np.nan

            
        # print(X_test_preprocessed)
            
        ### IMPUTATION & FEATURE SYNCHRONIZATION ###
        if X_test_preprocessed is not None:
            X_test_preprocessed = X_test[self.X_train_selected_cols]

            # check for full NaN columns all and fill first month with last obs from X_train
            # print("PREDICTION - X TEST CHECK FULL NANS") 
            # print(X_test_preprocessed.isnull().all(axis=0))
            # print(X_test_preprocessed.columns[X_test_preprocessed.isnull().all(axis=0)])

            nan_cols = X_test_preprocessed.columns[X_test_preprocessed.isnull().all(axis=0)].tolist()
            # print(nan_cols)

            # for nc in nan_cols:
            #     last_non_nan_index = self.X_train[nc].notna()[::-1].idxmax()
            #     print(last_non_nan_index)
            #     last_obs_from_X = self.X_train[nc][last_non_nan_index]
            #     print(last_obs_from_X)
            #     X_test_preprocessed[nan_cols] = self.X_train[nan_cols].loc[last_non_nan_indices, :]
                
            # print(self.X_train_imputed[nan_cols])
            # print(self.X_train_imputed[nan_cols].iloc[-1, :])
            
            # LOCF: fill nan cols with last known observation from training set
            X_test_preprocessed[nan_cols] = self.X_train_imputed[nan_cols].iloc[-1, :]
            # print(X_test_preprocessed[nan_cols])        

        # linear imputation
        if X_test_preprocessed is not None:
            X_test_preprocessed = X_test_preprocessed.interpolate(method="linear", axis=0, limit_direction="both")
            
        ### TRANSFORMATION: StandardScaler
        if X_test_preprocessed is not None and self.scaler is not None:
            X_test_preprocessed = self.scaler.transform(X_test_preprocessed)
        
        ### PCA
        if self.pca is not None and X_test is not None:
            X_test_preprocessed = self.pca.transform(X_test_preprocessed)
        else:
            X_test_preprocessed = None # if n_components==0
        
        ### ARIMA
        forecast = self.fitted_model.get_forecast(steps=fh, exog=X_test_preprocessed)
        y_pred = forecast.predicted_mean
        # set negative predictions to zero
        y_pred = y_pred.clip(lower=0)

        if ci == True:
            conf_int = forecast.summary_frame(alpha=0.1)
            return y_pred, conf_int
        
        else:
            return y_pred
    
    def predict_insample(self):
        
        forecast = self.fitted_model.predict(steps=self.y_train.shape[0], exog = self.X_train_preprocessed)
        y_hat = forecast
        
        return y_hat
