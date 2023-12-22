import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

import dataset.dataset as dataset
from dataset import TabularDataFrame
from model import get_classifier

from.utils import cal_metrics, set_seed

class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name
        
        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        self.feature_columns = dataframe.feature_columns
        self.target_column = dataframe.target_column
        self.train = dataframe.train
        self.test = dataframe.test
        self.id = dataframe.id

        self.input_dim = len(self.feature_columns)
        self.output_dim = len(np.unique(self.train[self.target_column]))

        self.seed = config.seed
        self.init_writer()
    
    def init_writer(self):
        metrics = [
            "fold",
            "ACC",
            "AUC"
        ]
        self.writer = {m: [] for m in metrics}

    
    def add_results(self, i_fold, scores: dict):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])
    
    def each_fold(self, i_fold, train_data, val_data):
        # uniq = self.get_unique(train_data)
        X, y = self.get_x_y(train_data)
        model_config = self.get_model_config()
        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        model.fit(
            X,
            y,
            eval_set=(val_data[self.feature_columns],
            val_data[self.target_column].values.squeeze()),
        )
        return model

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []

        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train[self.feature_columns], self.train[self.target_column])):
            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]

            model = self.each_fold(i_fold, train_data, val_data)

            score = cal_metrics(model, val_data, self.feature_columns, self.target_column)

            self.add_results(i_fold, score)

            y_test_pred_all.append(
                model.predict_proba(self.test[self.feature_columns])
            )
        print(f"AUC average: {sum(self.writer['AUC'])/len(self.writer['AUC'])}")
        y_test_pred_all = np.mean(y_test_pred_all, axis=0)
        submit_df = pd.DataFrame(self.id)

        submit_df[self.target_column] = y_test_pred_all
        print(submit_df)
        submit_df.to_csv("submit1.csv", index=False)
        
    
    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()
    
    # def get_unique(self, train_data):
    #     uniq = np.unique(train_data[self.target_column])
    #     return uniq
    
    def get_x_y(self, train_data):
        X, y = train_data[self.feature_columns], train_data[self.target_column].values.squeeze()
        return X, y
    
class ExpSimple(ExpBase):
    def __init__(self,config):
        super().__init__(config)
    
    def get_model_config(self, *args, **kwargs):
        return self.model_config
