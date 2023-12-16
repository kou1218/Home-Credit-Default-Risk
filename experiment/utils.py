import random
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
#one-v-oneで多クラスの性能評価
def cal_auc_score(model, data, feature_columns, target_column):
    pred_proba = model.predict_proba(data[feature_columns])
    auc = roc_auc_score(data[target_column], pred_proba)
    return auc

def cal_acc_score(model, data, feature_cols, target_column):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[target_column], pred)
    return acc

def cal_metrics(model, data, feature_cols, target_column):
    auc = cal_auc_score(model, data, feature_cols, target_column)
    acc = cal_acc_score(model, data, feature_cols, target_column)
    return {"ACC": acc, "AUC": auc}