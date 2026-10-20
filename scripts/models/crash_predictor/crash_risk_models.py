"""
Crash Risk Models – XGBoost classifier and SHAP explainer.
"""

import joblib
import xgboost as xgb
import shap


class PreRaceCrashModel:
    def __init__(self, **kwargs):
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": 1,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        default_params.update(kwargs)
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None

    def train(self, X, y, scale_pos_weight=None):
        if scale_pos_weight is not None:
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        return self.model

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"✅ Model loaded from {path}")