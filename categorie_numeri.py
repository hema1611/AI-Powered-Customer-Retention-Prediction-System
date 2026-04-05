import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
import os
import logger_config
import sys
import seaborn as sns
from logger_config import setup_logging
logger = setup_logging("categorie_numeri")
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder


def ct_t_nu(X_train_cat, X_test_cat):
    try:
        X_train_cat = X_train_cat.drop(['customerID'] , axis = 1)
        X_test_cat = X_test_cat.drop(['customerID'], axis = 1)
        logger.info(f'Before X_train_cat column : {X_train_cat.shape} : \n {X_train_cat.columns}')
        logger.info(f'Before X_test_cat column : {X_test_cat.shape} : \n {X_test_cat.columns}')
        # for col in X_train_cat.columns:
        #     logger.info(f"{col} unique values: {X_train_cat[col].unique()}")
        # print(X_train_cat)
        oh = OneHotEncoder(drop='first')
        oh.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'network_provider']])

        values_train = oh.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'network_provider']]).toarray()

        values_test = oh.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'network_provider']]).toarray()
        t1 = pd.DataFrame(values_train)
        t2 = pd.DataFrame(values_test)
        t1.columns = oh.get_feature_names_out()
        t2.columns = oh.get_feature_names_out()
        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, t1], axis=1)
        X_test_cat = pd.concat([X_test_cat, t2], axis=1)
        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'network_provider'], axis=1)
        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod', 'network_provider'], axis=1)
        logger.info(f"After Nominal X_train_cat Column : {X_train_cat.shape} : \n : {X_train_cat.columns}")
        logger.info(f"After Nominal X_test_cat Column : {X_test_cat.shape} : \n : {X_test_cat.columns}")

    #------------------------ordinal-----------------------
        od = OrdinalEncoder()
        od.fit(X_train_cat[['Contract']])
        results_train = od.transform(X_train_cat[['Contract']])
        results_test = od.transform(X_test_cat[['Contract']])
        p1 = pd.DataFrame(results_train)
        p2 = pd.DataFrame(results_test)
        p1.columns = od.get_feature_names_out() + "_od"
        p2.columns = od.get_feature_names_out() + "_od"
        p1.reset_index(drop=True, inplace=True)
        p2.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, p1], axis=1)
        X_test_cat = pd.concat([X_test_cat, p2], axis=1)
        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)
        logger.info(f"After ordinal X_train_cat Column : {X_train_cat.shape} : \n : {X_train_cat.columns}")
        logger.info(f"After ordinal X_test_cat Column : {X_test_cat.shape} : \n : {X_test_cat.columns}")

        logger.info(f"Train null values: {X_train_cat.isnull().sum()}")
        logger.info(f"Test null values: {X_test_cat.isnull().sum()}")

        return X_train_cat,X_test_cat
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')
