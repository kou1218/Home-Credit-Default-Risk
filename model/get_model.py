from experiment.utils import set_seed

from .gbm import XGBoostClassifier

def get_classifier(name, *, input_dim, output_dim, model_config, verbose=0, seed=42):
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose)
    else:
        raise KeyError(f"{name} is not defined.")