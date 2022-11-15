########################################
#              IMPORTS                 #
########################################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# supression of Warnings for all parallel jobs
import os
os.environ['PYTHONWARNINGS']='ignore'

import pandas as pd
import numpy as np
# random seed for numpy --> random seed for sklearn, pandas, statsmodels
np.random.seed(99)

import random
random.seed(99)

import matplotlib.pyplot as plt

# from sktime.forecasting.model_selection import (
#     SingleWindowSplitter,
#     SlidingWindowSplitter,
#     ExpandingWindowSplitter
# )

# Parallelization
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# Warninigs and Exceptions
from warnings import catch_warnings
import warnings
warnings.simplefilter('ignore')

import traceback

from tqdm import tqdm_notebook


from statsmodels.tools.sm_exceptions import ConvergenceWarning
# warnings.simplefilter('ignore', ConvergenceWarning)

from sklearn.exceptions import DataConversionWarning
# warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from forecasters import ParametricTimeSeriesForecaster

########################################
#         HELPER FUNCTIONS             #
########################################

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


def train_test_split(y, X, forecast_horizon):
    """
    Splits dfs y and X into training and test data set
    ----------
    y : pd.DataFrame with endogenous variables
    X : pd.DataFrame with exogenous variables
    forecast_horizon: number of observation in y_test and X_test
    """
    # rfcv = pm.model_selection.RollingForecastCV

    # 1 year aheaad forecast --> 12 months, 4 quarters = forecast_horizon
    y_train = y.iloc[0:-forecast_horizon]  
    y_test = y.iloc[-forecast_horizon:]

    if X is None:
        return None, None, y_train, y_test
    else:
        X_test = X.iloc[-forecast_horizon:]
        X_train = X.iloc[0:-forecast_horizon]

    return X_train, X_test, y_train, y_test


########################################
#         DATA PREPARATION             #
########################################

# returns df containing the time of the variable var_name of the countries in the list 
def get_series(data, target_variable="SUM(FATALITIES)", gid0=["MLI"]):
    # print("get_series")
    if gid0 is None:
        return None
    elif target_variable == "all":
        selection = [col for col in data.columns if col.endswith(tuple(gid0))] # selects all columns that end with ISO country code
    else:
        # selection = ["EVENT_DATE_MONTH"]
        selection = list()
        for g in gid0:
            selection.append(target_variable+"_"+g)
    return data[selection]

def getData(target_variable = "SUM(FATALITIES)",
            target_country = "MLI",
            predictor_countries = ["BFA"],
            socio_eco_vars = True,
            n_lags_X = 1,
            seasonal_periodicity = 12): # default: monthly data
    
    """
    returns a y and X pd.Dataframe with the specified data for conflict prediction.
    
    """
    print("Getting Data for "+ target_country)
    # load datasets
    if seasonal_periodicity == 12:
        acled_monthly_adm0_piv = pd.read_csv("../data/TB011_ACLED_MONTHLY_ADM0_LOG.csv",low_memory=False)
        imf_wb = pd.read_csv("../data/TB014_SOCIO_ECONOMIC_INDICATORS_RAW_MONTHLY.csv", low_memory=False)
        # print(acled_monthly_adm0_piv)
        # print(imf_wb)

    elif seasonal_periodicity == 4:
        acled_monthly_adm0_piv = pd.read_csv("../data/TB012_ACLED_QUARTERLY_ADM0_LOG.csv",low_memory=False)
        imf_wb = pd.read_csv("../data/TB014_SOCIO_ECONOMIC_INDICATORS_RAW_QUARTERLY.csv", low_memory=False)
        # print(acled_monthly_adm0_piv)
        # print(imf_wb)
        
    else:
        acled_monthly_adm0_piv = None
        print("ACLED IS NONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None
        
    print("IMF, WB, ACLED loaded")

    imf_wb = imf_wb.set_index(pd.DatetimeIndex(imf_wb.DATE), drop=True)
    imf_wb = imf_wb.drop(["DATE"], axis=1)
    
    # TODO: add and preserve datetime indexs
    acled_monthly_adm0_piv = acled_monthly_adm0_piv.set_index(pd.DatetimeIndex(acled_monthly_adm0_piv["EVENT_DATE_MONTH"]))

    # acled_monthly_adm0 = acled_monthly_adm0.set_index(pd.DatetimeIndex(acled_monthly_adm0["EVENT_DATE_MONTH"]))

    # select data for prediction
    try:
        # dependent variable y
        y = get_series(gid0 = [target_country], target_variable = target_variable, data = acled_monthly_adm0_piv)
       
        # independent predictor variables: neighbor time series & socio_economic indicators        
        neighbor_series = get_series(gid0 = predictor_countries, target_variable = "all", data = acled_monthly_adm0_piv)
        #ACLED Neighbor differencing
#         neighbor_series.iloc[:, :5].plot()
#         neighbor_series = neighbor_series.diff(periods=1)
#         neighbor_series.iloc[:, :5].plot()
        
        socio_eco = get_series(gid0=[target_country], target_variable="all", data=imf_wb)

        # print("------------NEIGHBOR--------------")
        # print(neighbor_series)
        # print("-------------SOCIO----------------")
        # print(socio_eco)
        if neighbor_series is None and socio_eco_vars:
            X = socio_eco
        elif neighbor_series is not None and not socio_eco_vars:
            X = neighbor_series
        elif neighbor_series is None and not socio_eco_vars:
            X = None
        else:
            X = pd.merge(neighbor_series, socio_eco, left_index=True, right_index=True, how="left") # left join drops all data before ACLED start dates
        # print(X)
        
    except ValueError as ve:
        print("++++++++++ getData(): EXCEPTION THROWN - VALUE ERROR++++++++++++++++")
        print(ve)
        print(traceback.format_exc())
        return None

    print("adjusting y for lags")
    ######## adjust y for lags ########
    # y_lagged = pd.DataFrame(y)
    # y_lagged = pd.DataFrame(y_lagged.iloc[n_lags_X:, 0]).reset_index(drop=True)
    y_lagged = pd.DataFrame(y.iloc[n_lags_X:, 0])

    ######## adjust X for lags ########
    # X_lagged = pd.DataFrame()
    
    # X = X.reset_index()
    
    print("adjusting X for lags")

    if X is not None:
        X_lagged = {"MONTH": X.index}    

        for var in X.columns:
            for n in range(1, n_lags_X+1): # 1...n_lags_X
                # print(n_lags_X-n, ":", -n)
                # print(var)
                col_name = str(var+"_T-"+str(n))
                # print(col_name)
                # print(np.array(X).flatten().shape)
                # X_lagged[col_name] = np.array(X).flatten()[n_lags_X-n:-n] # (2,-1)(1,-2)(0,-3)
                # print(X[var].shift(n, freq="MS"))
                if seasonal_periodicity == 12:
                    lagged_values = np.array(X[var].shift(n, freq="MS")) # lag X values
                elif seasonal_periodicity == 4:
                    lagged_values = np.array(X[var].shift(n, freq="3MS")) # lag X values
                else:
                    lagged_values = np.array(X[var].shift(n, freq="MS")) # lag X values

                # print("lagged_values")
                # print(lagged_values)
                X_lagged[col_name] = lagged_values
                # print(X[target_variable+"_"+c])
                # print(X_lagged[col_name])
                # print("X_LAGGED")
                # print(X_lagged)
                # print("--------------------------------------------------")

        # print(len(X_lagged))
        X_lagged = pd.DataFrame(X_lagged)
        X_lagged = X_lagged.set_index(pd.DatetimeIndex(X_lagged.MONTH))
        X_lagged = X_lagged.drop(["MONTH"], axis=1)
    else: 
        X_lagged = None
    # print(X_lagged)
    ####################################################################################################### 
    
    y_lagged["MONTH"] = y_lagged.index
    # y_lagged = y_lagged.set_index(pd.DatetimeIndex(y_lagged.MONTH))
    if seasonal_periodicity == 4:
        y_lagged = y_lagged.set_index(pd.PeriodIndex(y_lagged.MONTH, freq="3M"))
        if X_lagged is not None:
            X_lagged = X_lagged.set_index(pd.PeriodIndex(X_lagged.index, freq="3M"))

    else:
        y_lagged = y_lagged.set_index(pd.PeriodIndex(y_lagged.MONTH, freq="M"))
        if X_lagged is not None:
            X_lagged = X_lagged.set_index(pd.PeriodIndex(X_lagged.index, freq="M"))

    # drop month column
    y_lagged = y_lagged.drop(["MONTH"], axis=1)
    
    # drop all NaN rows from y
    y_lagged = y_lagged.dropna()

    print("GET DATA finished.")

    # let X start at same month as y
    start_y = y_lagged.index[0]
    if X_lagged is not None:
        X_lagged = X_lagged[start_y:]
        print("X: "+ str(X_lagged.shape))

    

    print("y: "+ str(y_lagged.shape))
    
    # print(y_lagged)
    
    print("------------------------------------------------")

    return y_lagged, X_lagged



class SequentialFeatureSelection:
    
    def __init__(self, model):
        self.model = model
        self.selected_columns = None
    
    def fit(self, X_train, y_train):
        
        self.selected_columns = 1
        return self.selected_columns
    
    def transform(self, X):
        pass
    
    

########################################
#           GRID SEARCH CV             #
########################################    

class GridSearchCV:
    
    def __init__(self, param_grid, forecaster: ParametricTimeSeriesForecaster, seasonal_periodicity = 12, cv_folds = 5):
        
        self.param_grid = param_grid
        self.forecaster = forecaster
        
        self.is_fitted = False
        
        self.cv_folds = cv_folds
        self.cv_results = list()
        self.best_model_cv_results = {}
        self.best_params = None
        self.best_model = None
        
        self.X_train = None
        self.y_train = None
                
        # parameters to allow model variety
        self.seasonal_periodicity = seasonal_periodicity
        
        self.MIN_OBS_CV = 2 * self.seasonal_periodicity # at least seasonal_periodicity obs. for training and seasonal_periodicity for validation

#         self.MIN_OBS_CV = 3 * self.seasonal_periodicity # at least 2*seasonal_periodicity obs. for training and 1*seasonal_periodicity for validation

    
    def train(self, X_train, y_train):
        print("Training started.")
        
        if len(y_train) < self.MIN_OBS_CV:
            # too little data for grid search
            print("GridSearchCV failed. Insufficient data for cross validation.")
            return None
        
#         elif np.var(y_train)[0] == 0:
#             print("y_train is constant. No GSCV necessary.")
#             return None
            
        else:
            print("Start GRIDSEARCH CV on "+str(cpu_count())+" CPU cores:")
            print(str(len(self.param_grid))+" parameter combinations are tested.")
            ### SUFFICIENT DATA --> START GRID_SEARCH_CV
            
            # print("---------------------- X_TRAIN------------------")
            # print(X_train)
            # print("---------------------- Y_TRAIN------------------")
            # print(y_train)
            self.X_train = X_train
            self.y_train = y_train
            self.y_t0 = y_train.iloc[-1, 0] # get last known value for log-change calculation

            ### parallel training and validation of all model specifications in grid
            # execute model CVs in parallel
            # executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            executor = Parallel(n_jobs=cpu_count(), backend='loky') # alternative backend: loky
#             executor = Parallel(n_jobs=1, backend='loky') # alternative backend: loky

            ############################### TODO: increase min train is
            # calculate number of folds
            MAX_FOLDS = self.cv_folds
            
            folds = min(int(len(y_train) / self.seasonal_periodicity) - 1, MAX_FOLDS) # save one fold for validation;
            cv_cutoffs = []
            for f in range(1, folds+1):
                new_cutoff = len(y_train)-1 - f*self.seasonal_periodicity
                cv_cutoffs.append(new_cutoff)
            cv_cutoffs.reverse()

            # list of cross validation results
            tasks = (delayed(self.cross_validate)(X_train, y_train, params, cv_cutoffs) for params in tqdm_notebook(self.param_grid))

            # save cv results
            self.cv_results = executor(tasks) 
            print("\n")

            # save results of best model = smallest avg validation score (TADDA)
            if len(self.cv_results) != 0:
                self.best_model_cv_results = min(self.cv_results, key=lambda x:x['cv_score'])

                params = self.best_model_cv_results["params"]
                print("Best CV-Results")
                print(self.best_model_cv_results)

                # retrain model on full training set        
                # try:            

                ### MODEL TRAINING
                self.best_model = self.forecaster()
                # print(X_train.shape)
                # print(y_train.shape)

                # print(X_train)
                # print(y_train)
                self.best_model_result = self.best_model.fit(X_train, y_train, params)
                if self.best_model_result is not None:
                    #########################################
                    # except:
                    #     print("RETRAINING - EXCEPTION DURING FITTING")    
                    self.is_fitted = True
                    print("Training finished.")

                    return self.best_model_result.summary()
                else:
                    print("Retraining model failed.")
                    return None

            else:
                print("GridSearchCV failed. No optimal model found.")
                return None

        
    def cross_validate(self, X_train, y_train, params, cv_cutoffs):
        
#         print("-------------------- PARAMS: "+str(params)+ " -------------")
        
        # Slinding Window CV ############ TODO: QUARTERLY
        
        # initial_window = int(round(len(y_train)*0.5))        
        
        # ews = ExpandingWindowSplitter(fh = fh, 
        #                               initial_window = fold_size, 
        #                               step_length = fold_size)
        # sws = SingleWindowSplitter(fh = fh, window_size = round(len(y_train)*0.7))
        # slws = SlidingWindowSplitter(fh = fh, initial_window = round(len(y_train)*0.5), step_length = 6)
        
        ### EXPANDING WINDOW SPLITTER
#         if len(y_train) >= 2*FH and len(y_train) <= 3*FH-1:
#             cv_cutoffs = [len(y_train)-1-FH]
#         elif len(y_train) >= 3*FH and len(y_train) <= 4*FH-1:
#             cv_cutoffs = [len(y_train)-1-FH]
#         elif len(y_train) >= 4*FH and len(y_train) <= 5*FH-1:
#             cv_cutoffs = [len(y_train)-1-FH]
#         elif len(y_train) >= 5*FH and len(y_train) <= 6*FH-1:
#             cv_cutoffs = [len(y_train)-1-FH]
#         elif len(y_train) >= 6*FH:
#             cv_cutoffs = [len(y_train)-1-FH]
            
        
        
#         print("LENGTH Y TRAIN"+str(len(y_train)))
#         print("FOLDS" + str(folds))
        
        
            
        # print("CV_CUTOFFS:" + str(cv_cutoffs))
        # cv_cutoffs = ews.get_cutoffs(y_train)
        # cv_cutoffs = [round(len(y_train)*0.7)]
        
        FH = self.seasonal_periodicity # 12 for monhtly predictions; 4 for quarterly predictions 

        fold_results = []
        for fold, cutoff in enumerate(cv_cutoffs):
            
#             print("-------------------- FOLD: "+str(fold)+ " -------------")
            # split training set --> cv training set + cv validation set
            if X_train is not None:
                X_train_cv = X_train.iloc[:cutoff]
                X_val_cv = X_train.iloc[cutoff:cutoff+FH]
                            
                ### remove constant columns from training data
                X_train_cv = X_train_cv.loc[:, (X_train_cv != X_train_cv.iloc[0]).any()] 
            
                ### remove same columns from validation data --> same columns as training data
                X_val_cv = X_val_cv.loc[:, X_train_cv.columns]
                
                if X_val_cv.empty or X_train_cv.empty:
                    X_train_cv = None
                    X_val_cv = None
            else:
                X_train_cv = None
                X_val_cv = None
                
            y_train_cv = y_train.iloc[:cutoff]
            y_val_cv = y_train.iloc[cutoff:cutoff+FH]
            
            # try:
            
#             print("CV SPLITING----------------")
#             print(0, cutoff)
#             print(cutoff, cutoff+FH)

#             print("-------------------------------------")

            model = self.forecaster()
            model_fit = model.fit(X_train_cv, y_train_cv, params)
            
            if model_fit is not None:
                ### 1-YEAR-AHEAD PREDICITON
                y_pred = model.predict(X_test = X_val_cv, fh = len(y_val_cv))

                ### OUT-OF-SAMPLE MODEL EVALUATION VIA TADDA-SCORE         
                eval_df = pd.DataFrame()

                y_pred_logchange = y_pred - y_train_cv.iloc[-1, 0]
                y_val_cv_logchange = y_val_cv - y_train_cv.iloc[-1, 0]

                eval_df["LC(FAT_PRED)"] = y_pred_logchange.values
                eval_df["LC(FAT_ACTUAL)"] = y_val_cv_logchange.values
                # print(eval_df)

                tadda = np.mean(eval_df.apply(lambda x: tadda_error(x["LC(FAT_ACTUAL)"], x["LC(FAT_PRED)"]), axis = 1))
                            
                # print("FOLD: "+str(fold)+", TADDA: "+ str(tadda))
            else:
                # model fit failed due to e.g. insuffienct data
                tadda = np.nan

            ### RESULT DICTIONARY
            fold_results.append(tadda)
            #########################################
            # except:
            #     print("CROSS-VALIDATION: EXCEPTION DURING FITTING")
        
        # calculate final validation score as average ovér folds
        avg_score = np.nanmean(fold_results) # ignoring nans
        # print("PARAMS: "+str(params)+"\t MEAN(TADDA): "+str(avg_score))
        
        result_dict = {
            "params": params,
            "cv_score": avg_score,
            "fold_results": fold_results
        }
        # print(result_dict)
        # print("#", end="")
        
        return result_dict
    
    def evaluate_model(self, X_test, y_test, fh):
        print("Evaluation started.")
        
        if self.is_fitted is True: # check if model is trained
            
            ### 1-YEAR-AHEAD PREDICITON
            # Based on the nature of the ARIMA equations, out-of-sample forecasts tend to converge to the sample mean for long forecasting periods.
            # print("Based on the nature of the ARIMA equations, out-of-sample forecasts tend to converge to the sample mean for long forecasting periods.")
        
            forecast, conf_int = self.best_model.predict(X_test = X_test, 
                                                         fh = len(y_test), 
                                                         ci = True)
           
            # CONFIDENCE INTERVALS
            # print(conf_int)
            
            # print(X_test, y_test)
            forecast_logchange = forecast - self.y_t0

            ### LOG-CHANGE CALCULATION
            y_test_logchange = y_test - self.y_t0
            
            ### PLOT IN-SAMPLE FITTED VALUES
            fig, ax = plt.subplots(figsize=(15, 10))
            
            self.y_train.plot(ax=ax, label="y_train", marker="o", color="blue")
            y_test.plot(ax=ax, label="y_test", marker="o", color="lightblue")

            y_hat = self.best_model.predict_insample()
            # print(y_hat)
            y_hat.plot(ax=ax, label="y_hat", marker="o", color="red")

            ### PLOT OUT-OF-SAMPLE PREDICTION RESULTS
            forecast.plot(label="predicted mean", marker="o", color="red")

            ax.fill_between(conf_int.index, conf_int["mean_ci_lower"], conf_int["mean_ci_upper"], alpha=0.1, label="90%-confidence intervall", color="red")
            
            plt.title("In-sample fit and out-of-sample prediction (log. abs. values): " + str(y_test.columns[0]))
            plt.grid()
            plt.legend(["y_train ", "y_test", "y_hat", "predicted mean", "90%-confidence intervall"])
            plt.show()
            
            ### LOG-CHANGE FORECAST VISUALIZATION
            fig, ax = plt.subplots(figsize=(15, 10))
            y_test_logchange.plot(ax=ax, marker="o", color="lightblue") 
            forecast_logchange.plot(ax=ax, marker="o", color="red") 

            plt.title("Out-of-sample predictions (log-change): " + str(y_test_logchange.columns[0]))
            plt.legend(["y_test", "predicted mean"])
            plt.grid()
            plt.show()
            
            ### BUILD PREDICITON RESULT DF FOR MODEL EVALUATION
            table = pd.DataFrame()
            table["MONTH"] = y_test.index
            table["FAT_PRED"] = np.exp(forecast.to_numpy()) - 1 # absolute prediction
            table["FAT_ACTUAL"] = np.exp(y_test.to_numpy()) - 1 # absolute observed fatalities
            # log change
            table["LC(FAT_PRED)"] = forecast_logchange.to_numpy()
            table["LC(FAT_ACTUAL)"] = y_test_logchange.to_numpy()
 
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