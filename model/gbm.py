from .base_model import BaseClassifier

import xgboost as xgb

from .utils import f1_micro

class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=self.output_dim,
            eval_metric=f1_micro,
            early_stoppping_rounds=50,
            **self.model_config,
        )
    
    def fit(self, X, y, eval_set):
        self.model.fit(X, y, eval_set=[eval_set], verbose=self.verbose)
        
    
    def predict_proba(self, X):
        pred = self.model.predict_proba(X)
        return pred[:, 1]

    def predict(self, X):
        return self.model.predict(X.values)
    
    def evalueate(self, X, y):
        raise NotImplementedError()