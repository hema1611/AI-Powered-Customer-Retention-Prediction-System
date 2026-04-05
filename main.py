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
logger = setup_logging("main")
from sklearn.model_selection import train_test_split
from mode_imputation import missing_values
from var_trans_outliers import vt
from filter_methods import filter_mt
from categorie_numeri import ct_t_nu
from imblearn.over_sampling import SMOTE
from feature_scaling import fs

class SIM:
    def __init__(self, path):
        try:
            # loding the data
            self.path = path
            self.df = pd.read_csv(self.path)
            # basic info
            logger.info(f"Total data size: {self.df.shape}")
            logger.info(f"null values : {self.df.isnull().sum()}")
            # adding network provider column (randomly)
            # created the network provider without changing original
            self.df['network_provider'] = self.df['PaymentMethod'].map({'Electronic check': 'Airtel',
                                                                        'Bank transfer (automatic)':'jio',
                                                                        'Credit card (automatic)': 'vi',
                                                                        'Mailed check' :'bsnl'})
            logger.info(f'Total data size: {self.df.shape}')
            # logger.info(f'====================== After adding the Network row =================')
            logger.info(f"Sample data :\n {self.df}")

            # In the column Total charges the data is in numerical but it showing in object type
            # so any extra spaces removed by nan and converted to numerical

            self.df['TotalCharges'] = self.df['TotalCharges'].replace("", np.nan)
            logger.info(f"total null values : \n  {self.df.isnull().sum()}")
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            logger.info(f' After changing the object to numeric the null values are :  \n {self.df.isnull().sum()}')
            logger.info(f'Total data size: \n {self.df.shape}')
            #total null values are 11 from the column total charges

            self.X = self.df.drop('Churn', axis = 1)  # independent col
            self.y = self.df['Churn'] #dependent col

            # spliting the data into training and testing

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            self.y_train = self.y_train.map({'Yes' : 1, 'No' : 0 }).astype(int)
            self.y_test = self.y_test.map({'Yes' : 1, 'No' : 0 }).astype(int)
            logger.info(f'Train divided Data size : {len(self.X_train)} : {len(self.y_train)}')
            logger.info(f'Test divided Data size : {len(self.X_test)} : {len(self.y_test)}')
            logger.info(f'total train data shape : {self.X_train.shape}')
            logger.info(f'total test data shape: {self.X_test.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')


    def handling_missing_values(self):
        try:
            self.X_train, self.X_test = missing_values(self.X_train, self.X_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

    def data_separation(self):
        try:
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')


            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f'{self.X_train_num.columns} : {self.X_train_num.shape}')
            logger.info(f'{self.X_test_num.columns} : {self.X_test_num.shape}')
            logger.info(f'=================================================================')
            logger.info(f'{self.X_train_cat.columns} : {self.X_train_cat.shape}')
            logger.info(f'{self.X_test_cat.columns} : {self.X_test_cat.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

    def variable_transformation(self):
        try:
        #     variable_outliers(self.X_train_num_cols, self.X_test_num_cols)
        #     plt.figure(figsize=(5,3))
        #     for i in self.X_train_num_cols.columns:
        # # ------------------------ to know the outliers ---------------------
        #         sns.boxplot(x = self.X_train_num_cols[i])
        #         plt.show()

            self.X_train_num,self.X_test_num = vt(self.X_train_num, self.X_test_num)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

    def feature_selection(self):
        try:
            self.X_train_num,self.X_test_num = filter_mt(self.X_train_num , self.X_test_num,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

    def cat_to_num(self):
        try:
            self.X_train_cat, self.X_test_cat = ct_t_nu(self.X_train_cat, self.X_test_cat)
        # ---------------------------combine the data------------------------------------
            self.X_train_num.reset_index(drop = True, inplace=True)
            self.X_train_cat.reset_index(drop = True, inplace=True)
            self.X_test_num.reset_index(drop = True, inplace=True)
            self.X_test_cat.reset_index(drop = True, inplace=True)

            self.training_data = pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_data = pd.concat([self.X_test_num,self.X_test_cat],axis=1)

            logger.info(f'------------------------------------------------')
            logger.info(f'Final Training data : {self.training_data.shape}')
            logger.info(f'{self.training_data.columns}')
            logger.info(f'{self.training_data.isnull().sum()}')

            logger.info(f'Final Testing data : {self.testing_data.shape}')
            logger.info(self.testing_data.columns)
            logger.info(f'{self.testing_data.isnull().sum()}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')

    def data_balancing(self):
        try:
            logger.info(f'Number of Rows for good customer {1} : {sum(self.y_train == 1)}')
            logger.info(f'Number of Rows for Bad customer {0} : {sum(self.y_train == 0)}')
            logger.info(f'Before training data shape : {self.training_data.shape}')

            sm = SMOTE(random_state=42)

            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data,self.y_train)
            logger.info(f'Number of Rows for good customer {1} : {sum(self.y_train_bal == 1)}')
            logger.info(f'Number of Rows for Bad customer {0} : {sum(self.y_train_bal == 0)}')
            logger.info(f'After training data shape : {self.training_data_bal.shape}')

            fs(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')



if __name__ == "__main__":
    try:
        obj = SIM('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.handling_missing_values()
        obj.data_separation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in Line no : {er_line.tb_lineno} due to : {er_msg}')
