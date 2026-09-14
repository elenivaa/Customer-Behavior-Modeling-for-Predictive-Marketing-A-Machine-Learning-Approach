import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


def build_and_train_churn_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> ImbPipeline:
    """
    Constructs and fits a production-grade, leak-free pipeline:
    LogTransform -> StandardScaler -> SMOTE -> XGBClassifier
    """
    pipeline = ImbPipeline([
        ("log_transform", FunctionTransformer(np.log, validate=False)),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
        ("classifier", XGBClassifier(
            max_depth=3,
            learning_rate=0.01,
            n_estimators=50,
            random_state=random_state,
            eval_metric="logloss"
        ))
    ])
    
    print("Fitting production pipeline...")
    pipeline.fit(X_train, y_train)
    return pipeline


def load_pipeline(filepath: str = "Results/churn_prediction_pipeline.joblib") -> ImbPipeline:
    """Loads a serialized joblib pipeline artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pipeline artifact not found at {filepath}")
    return joblib.load(filepath)
