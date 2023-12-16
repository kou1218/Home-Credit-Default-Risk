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
        categorical_encoder="ordinal",
        continuous_encoder: str = None,
        **kwargs,
    ) -> None:

        self.seed = seed
        self.categorical_encoder = categorical_encoder
        self.continuous_encoder = continuous_encoder

        self.train = pd.read_csv(to_absolute_path('input/train.csv'))
        self.test = pd.read_csv(to_absolute_path('input/test.csv'))
        self.id = self.test['SK_ID_CURR']

        self.train = self.train[self.feature_columns + [self.target_column]]
        self.test = self.test[self.feature_columns]


        #クラスがobjectの場合
        # self.label_encoder = LabelEncoder().fit(self.train[self.target_column])
        # self.train[self.target_column] = self.label_encoder.trainsform(self.tarin[self.target_column])

    
    
    def processed_dataframes(self) -> Dict[str, pd.DataFrame]:
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
    feature_columns = continuous_columns

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)



#     # self.train = pd.read_csv()
#     # self.test = pd.read_csv()

#     # self.id = self.test["SK_ID_CURR"]

#     # self.train.drop(columns=[""], inplace=True)
#     # self.test.drop(columns=[""], inplace=True)