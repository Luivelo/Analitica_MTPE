import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures, FunctionTransformer, TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, RFE, RFECV, SelectFromModel, mutual_info_regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, VotingRegressor, AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from imblearn.over_sampling import SMOTE
from typing import List, Tuple, Any, Dict, Literal
from scipy.stats import uniform, randint
from collections import Counter
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')




# MODELO PARA PREDECIR GRAVEDAD DE ACCIDENTES
class TreeEnsemblePipeline:
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.feature_selector = SelectKBest(k=self.n_features)
        self.scaler = StandardScaler()
        self.model = None
                
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        return X_smote, y_smote
            
    def create_ensemble(self) -> VotingClassifier:
        dt = DecisionTreeClassifier(random_state=42)
                
        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
                
        histgb = HistGradientBoostingClassifier(
            random_state=42,
            validation_fraction=0.1,
            early_stopping=True,
            max_iter=1000
        )
                
        catboost = CatBoostClassifier(
            random_state=42,
            verbose=False,
            allow_writing_files=False
        )
                
        adaboost = AdaBoostClassifier(
            random_state=42
        )
                
        rf = RandomForestClassifier(
            random_state=42
        )
                
        ensemble = VotingClassifier(
            estimators=[
                ('dt', dt),
                ('xgb', xgb),
                ('histgb', histgb),
                ('catboost', catboost),
                ('adaboost', adaboost),
                ('rf', rf)
            ],
            voting='soft'
        )
        return ensemble

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomizedSearchCV, pd.DataFrame, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
                
        X_train, y_train = self.apply_smote(X_train, y_train)
                
        ensemble = self.create_ensemble()
                
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('selector', self.feature_selector),
            ('model', ensemble)
        ])
                
        param_dist = {
            'model__dt__max_depth': randint(3, 7),
            'model__dt__min_samples_split': randint(2, 10),
                    
            'model__xgb__n_estimators': randint(100, 200),
            'model__xgb__max_depth': randint(3, 7),
            'model__xgb__learning_rate': uniform(0.01, 0.1),
                        
            'model__histgb__max_depth': randint(3, 7),
            'model__histgb__learning_rate': uniform(0.01, 0.1),
            'model__histgb__min_samples_leaf': randint(10, 20),
                    
            'model__catboost__depth': randint(4, 8),
            'model__catboost__learning_rate': uniform(0.01, 0.1),
            'model__catboost__iterations': randint(100, 200),
                    
            'model__adaboost__n_estimators': randint(50, 150),
            'model__adaboost__learning_rate': uniform(0.01, 0.1),
                    
            'model__rf__n_estimators': randint(100, 200),
            'model__rf__max_depth': randint(3, 7),
            'model__rf__min_samples_split': randint(2, 10)
        }
                
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=50,  # Número de combinaciones a probar
            cv=3,  # Número de folds en la validación cruzada
            scoring='roc_auc',  # Optimizar para auc roc
            n_jobs=-1,  # Usar todos los cores disponibles
            random_state=42
        )
                
        random_search.fit(X_train, y_train)
        self.model = random_search
        return random_search, X_test, y_test

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been built yet. Call 'build_model' first.")
        return self.model.predict(X)