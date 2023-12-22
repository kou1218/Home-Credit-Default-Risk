import numpy as np
import pandas as pd

from hydra.utils import to_absolute_path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from hydra.utils import to_absolute_path

from typing import Any, Dict, List

class TabularDataFrame(object):
    all_columns =[
        "SK_ID_CURR",
        "TARGET",            
        "NAME_CONTRACT_TYPE",                  
        "CODE_GENDER",                         
        "FLAG_OWN_CAR",                  
        "FLAG_OWN_REALTY",                 
        "CNT_CHILDREN",                        
        "AMT_INCOME_TOTAL",                   
        "AMT_CREDIT",                          
        "AMT_ANNUITY",                         
        "AMT_GOODS_PRICE",                   
        "NAME_TYPE_SUITE",                   
        "NAME_INCOME_TYPE",                    
        "NAME_EDUCATION_TYPE",                 
        "NAME_FAMILY_STATUS",                  
        "NAME_HOUSING_TYPE",                   
        "REGION_POPULATION_RELATIVE",          
        "DAYS_BIRTH",                         
        "DAYS_EMPLOYED",                       
        "DAYS_REGISTRATION",                  
        "DAYS_ID_PUBLISH",                    
        "OWN_CAR_AGE",                    
        "FLAG_MOBIL",                        
        "FLAG_EMP_PHONE",                      
        "FLAG_WORK_PHONE",                     
        "FLAG_CONT_MOBILE",                    
        "FLAG_PHONE",                        
        "FLAG_EMAIL",                         
        "OCCUPATION_TYPE",                 
        "CNT_FAM_MEMBERS",                     
        "REGION_RATING_CLIENT",                
        "REGION_RATING_CLIENT_W_CITY",         
        "REG_REGION_NOT_LIVE_REGION",          
        "REG_REGION_NOT_WORK_REGION",          
        "LIVE_REGION_NOT_WORK_REGION",         
        "REG_CITY_NOT_LIVE_CITY",          
        "REG_CITY_NOT_WORK_CITY",                      
        "ORGANIZATION_TYPE",                  
        "EXT_SOURCE_1",                   
        "EXT_SOURCE_2",                      
        "EXT_SOURCE_3",                    
        "OBS_30_CNT_SOCIAL_CIRCLE",          
        "DEF_30_CNT_SOCIAL_CIRCLE",          
        "OBS_60_CNT_SOCIAL_CIRCLE",          
        "DEF_60_CNT_SOCIAL_CIRCLE",          
        "DAYS_LAST_PHONE_CHANGE",          
        "AMT_REQ_CREDIT_BUREAU_HOUR",      
        "AMT_REQ_CREDIT_BUREAU_MON",       
        "AMT_REQ_CREDIT_BUREAU_QRT",    
        "AMT_REQ_CREDIT_BUREAU_YEAR",

    ]

    
    continuous_columns = []
    categorical_columns = []
    binary_columns = []
    #実際に使うcolumns
    feature_columns = []
    target_column = "TARGET"

    def __init__(
        self,
        seed,
        categorical_encoder: str = 'ordinal',
        continuous_encoder: str = None,
        **kwargs,
    ) -> None:

        self.seed = seed
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

        self.train = pd.read_csv(to_absolute_path('input/train.csv'))
        self.test = pd.read_csv(to_absolute_path('input/test.csv'))
        self.id = self.test['SK_ID_CURR']

        self.feature_columns = self.continuous_columns + self.categorical_columns

        self.train = self.train[self.feature_columns + [self.target_column]]
        self.test = self.test[self.feature_columns]

        self.processed_dataframes()
    
    def fit_con_encoder(self, df):
        con_data = df[self.continuous_columns]
        if(self.continuous_columns != []):
            if(self.continuous_encoder == 'minmax'):
                scaled_data = MinMaxScaler().fit_transform(con_data)
            elif(self.continuous_encoder == 'standard'):
                scaled_data = StandardScaler().fit_transform(con_data)
            else:
                raise ValueError(self.continuous_encoder)
            con_data = pd.DataFrame(scaled_data, columns=con_data.columns)
        return con_data

    def fit_cate_encoder(self, df):
        cate_data = df[self.categorical_columns]
        if(self.categorical_columns != []):
            if(self.categorical_encoder == 'ordinal'):
                scaled_data = OrdinalEncoder().fit_transform(cate_data)
            # elif(self.categorical_encoder == 'onehot'):

            else:
                raise ValueError(self.categorical_encoder)
            cate_data = pd.DataFrame(scaled_data, columns=cate_data.columns)

        return cate_data


    
    
    def add_features(self):
        ...
    
    def drop_features(self):
        ...

    def processed_dataframes(self) :
        self.train[self.continuous_columns] = self.fit_con_encoder(self.train[self.continuous_columns])
        self.test[self.continuous_columns] = self.fit_con_encoder(self.test[self.continuous_columns])
        self.train[self.categorical_columns] = self.fit_cate_encoder(self.train[self.categorical_columns])
        self.test[self.categorical_columns] = self.fit_cate_encoder(self.test[self.categorical_columns])
        print('ok!')
        ...
        # """
        # Returns:
        #     dict[str, DataFrame]: The value has the keys "train", "val" and "test".
        # """
        # dfs = self.get_classify_dataframe()
        # # preprocessing
        # self.fit_feature_encoder(dfs["train"])
        # dfs = self.apply_feature_encoding(dfs)
        # self.all_columns = list(self.categorical_columns) + list(self.continuous_columns) + list(self.binary_columns)
        # return df
    
    


class V0(TabularDataFrame):
   
    
    continuous_columns = [
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL',
       'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
       'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
    ]
    categorical_columns = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
    ]

    target_column = "TARGET"
    

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    # def processed_dataframes(self):

# class V1(TabularDataFrame):
