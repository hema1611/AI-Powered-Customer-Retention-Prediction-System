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
logger = setup_logging("var_trans_outliers")
from sklearn.preprocessing import QuantileTransformer

def vt(X_train_num,X_test_num):
    try:
        logger.info(f"Before Train Column Name : {X_train_num.columns}")
        logger.info(f"Before Test Column Name : {X_test_num.columns}")
        for i in X_train_num.columns:
            qt = QuantileTransformer(output_distribution='normal')

            X_train_num[i + '_qt'] = qt.fit_transform(X_train_num[[i]])
            X_test_num[i + '_qt'] = qt.transform(X_test_num[[i]])

            X_train_num = X_train_num.drop([i], axis=1)
            X_test_num = X_test_num.drop([i], axis=1)
            # trimming
            iqr = X_train_num[i + '_qt'].quantile(0.75) - X_train_num[i + '_qt'].quantile(0.25)
            lower_limit = X_train_num[i + '_qt'].quantile(0.25) - (1.5 * iqr)
            upper_limit = X_train_num[i + '_qt'].quantile(0.75) + (1.5 * iqr)
            X_train_num[i + '_trim'] = np.where(X_train_num[i + '_qt'] > upper_limit, upper_limit,
                                                np.where(X_train_num[i + '_qt'] < lower_limit, lower_limit,
                                                         X_train_num[i + '_qt']))
            X_test_num[i + '_trim'] = np.where(X_test_num[i + '_qt'] > upper_limit, upper_limit,
                                               np.where(X_test_num[i + '_qt'] < lower_limit, lower_limit,
                                                        X_test_num[i + '_qt']))

            X_train_num = X_train_num.drop([i + '_qt'], axis=1)
            X_test_num = X_test_num.drop([i + '_qt'], axis=1)
        logger.info(f"After Train Column Name : {X_train_num.columns}")
        logger.info(f"After Test Column Name : {X_test_num.columns}")
        return X_train_num, X_test_num
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

