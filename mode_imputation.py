import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import logging
import warnings
warnings.filterwarnings("ignore")
from logger_config import setup_logging
logger = setup_logging("mode_imputation")

def missing_values(X_train, X_test):
    try:
        logger.info(f'before handling null values X_train : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}')
        logger.info(f'before handling null values X_test : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}')

        for i in X_train.columns:
            if X_train[i].isnull().sum()>0:
                X_train[i+'_mode']=X_train[i].fillna(X_train[i].mode()[0])
                X_test[i+'_mode']=X_test[i].fillna(X_test[i].mode()[0])
                X_train = X_train.drop(['TotalCharges'],axis = 1)
                X_test = X_test.drop(["TotalCharges"], axis = 1)

        logger.info(f'After handling null values X_train : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}')
        logger.info(f'After handling null values X_test : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}')

        return X_train, X_test
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')