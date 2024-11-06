import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures, FunctionTransformer, TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, RFE, RFECV, SelectFromModel, mutual_info_regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, mean_squared_error, r2_score
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



@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    n_features: int = 12
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    n_iter_search: int = 30
    alpha: float = 0.8
    winsor_limits: tuple = (0.05, 0.95)

class Winsorizer(BaseEstimator, TransformerMixin):
    """Custom transformer for winsorization"""
    def __init__(self, limits=(0.05, 0.95)):
        self.limits = limits
        self.feature_bounds_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.feature_bounds_ = []
        for col in range(X.shape[1]):
            lower = np.percentile(X[:, col], self.limits[0] * 100)
            upper = np.percentile(X[:, col], self.limits[1] * 100)
            self.feature_bounds_.append((lower, upper))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for i, (lower, upper) in enumerate(self.feature_bounds_):
                X.iloc[:, i] = X.iloc[:, i].clip(lower=lower, upper=upper)
        else:
            X = X.copy()
            for i, (lower, upper) in enumerate(self.feature_bounds_):
                X[:, i] = np.clip(X[:, i], lower, upper)
        return X

class RMSECallback:
    """Callback to track RMSE during training"""
    def __init__(self):
        self.training_rmse = []
    
    def __call__(self, rmse):
        self.training_rmse.append(rmse)

class ModelWithCallback(BaseEstimator, RegressorMixin):
    """Wrapper class to add RMSE tracking to any model"""
    def __init__(self, base_model, callback=None):
        self.base_model = base_model
        self.callback = callback
    
    def fit(self, X, y):
        self.base_model.fit(X, y)
        if self.callback:
            y_pred = self.base_model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            self.callback(rmse)
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)

class OptimizedEnsemblePipeline:
    """Main pipeline class for model training and prediction"""
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.logger = self._setup_logger()
        self.feature_selector = SelectKBest(
            score_func=mutual_info_regression,
            k='all'
        )
        self.scaler = RobustScaler()
        self.rmse_callback = RMSECallback()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _analyze_data_distribution(self, y: pd.Series):
        """Analyze data distribution before and after winsorization"""
        try:
            self.logger.info("Original data statistics:")
            self._log_distribution_stats(y, "Original")
            
            # Analyze winsorized data
            winsorizer = Winsorizer(limits=self.config.winsor_limits)
            y_winsorized = pd.Series(
                winsorizer.fit_transform(y.values.reshape(-1, 1)).ravel()
            )
            self.logger.info("\nWinsorized data statistics:")
            self._log_distribution_stats(y_winsorized, "Winsorized")
            
        except Exception as e:
            self.logger.warning(f"Error in data distribution analysis: {str(e)}")

    def _log_distribution_stats(self, data: pd.Series, label: str):
        """Helper method to log distribution statistics"""
        self.logger.info(f"{label} statistics:")
        self.logger.info(f"Mean: {data.mean():.2f}")
        self.logger.info(f"Std: {data.std():.2f}")
        self.logger.info(f"Min: {data.min():.2f}")
        self.logger.info(f"Max: {data.max():.2f}")
        
        quartiles = data.quantile([0.25, 0.5, 0.75])
        self.logger.info(f"Quartiles:\n{quartiles}")
        
        IQR = quartiles[0.75] - quartiles[0.25]
        outlier_threshold = IQR * 1.5
        outliers = ((data < (quartiles[0.25] - outlier_threshold)) | 
                   (data > (quartiles[0.75] + outlier_threshold))).sum()
        self.logger.info(f"Number of potential outliers: {outliers}")

    def _create_base_models(self) -> Dict[str, BaseEstimator]:
        """Create base models with RMSE tracking"""
        base_models = {
            'gbm': GradientBoostingRegressor(
                random_state=self.config.random_state,
                n_estimators=100,
                learning_rate=0.05,
                subsample=0.8,
                max_depth=5,
                alpha=0.9
            ),
            'rf': RandomForestRegressor(
                random_state=self.config.random_state,
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                n_jobs=-1,
                bootstrap=True
            ),
            'xgb': XGBRegressor(
                random_state=self.config.random_state,
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                reg_alpha=self.config.alpha
            ),
            'histgb': HistGradientBoostingRegressor(
                random_state=self.config.random_state,
                max_iter=100,
                learning_rate=0.05,
                max_depth=5,
                l2_regularization=self.config.alpha
            )
        }
        
        return {name: ModelWithCallback(model, self.rmse_callback) 
                for name, model in base_models.items()}

    def _create_param_distributions(self) -> Dict[str, Any]:
        """Create parameter distributions for hyperparameter optimization"""
        return {
            'model__gbm__base_model__n_estimators': randint(50, 200),
            'model__gbm__base_model__learning_rate': uniform(0.01, 0.1),
            'model__gbm__base_model__max_depth': randint(3, 7),
            'model__gbm__base_model__subsample': uniform(0.7, 0.3),
            
            'model__rf__base_model__n_estimators': randint(50, 200),
            'model__rf__base_model__max_depth': randint(5, 15),
            'model__rf__base_model__min_samples_leaf': randint(1, 5),
            
            'model__xgb__base_model__n_estimators': randint(50, 200),
            'model__xgb__base_model__learning_rate': uniform(0.01, 0.1),
            'model__xgb__base_model__max_depth': randint(3, 7),
            'model__xgb__base_model__subsample': uniform(0.7, 0.3),
            'model__xgb__base_model__colsample_bytree': uniform(0.7, 0.3),
            
            'model__histgb__base_model__learning_rate': uniform(0.01, 0.1),
            'model__histgb__base_model__max_depth': randint(3, 7),
            'model__histgb__base_model__l2_regularization': uniform(0.5, 2.0)
        }

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomizedSearchCV, pd.DataFrame, pd.Series, List[float]]:
        """Build and train the model with separate winsorization for features and target"""
        try:
            self.logger.info(f"Starting model building process with {len(X)} samples...")
            
            # Split data first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Create two separate winsorizers for features and target
            feature_winsorizer = Winsorizer(limits=self.config.winsor_limits)
            target_winsorizer = Winsorizer(limits=self.config.winsor_limits)
            
            # Create ensemble with optimized weights
            base_models = self._create_base_models()
            ensemble = VotingRegressor(
                estimators=[(name, model) for name, model in base_models.items()],
                weights=[1.2, 1.0, 1.3, 1.1]
            )
            
            # Create pipeline with separate winsorization for features
            pipeline = Pipeline([
                ('feature_winsorizer', feature_winsorizer),
                ('scaler', self.scaler),
                ('selector', self.feature_selector),
                ('model', ensemble)
            ])
            
            # Winsorize the target variable separately
            y_train_winsorized = target_winsorizer.fit_transform(
                y_train.values.reshape(-1, 1)
            ).ravel()
            
            # Analyze distributions
            self._analyze_data_distribution(y_train)
            
            # Train and evaluate with winsorized target
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=self._create_param_distributions(),
                n_iter=self.config.n_iter_search,
                cv=self.config.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.config.random_state,
                error_score='raise'
            )
            
            self.logger.info("Fitting model...")
            random_search.fit(X_train, y_train_winsorized)
            
            # Store winsorizers for later use in predictions
            self.feature_winsorizer_ = feature_winsorizer
            self.target_winsorizer_ = target_winsorizer
            
            # Show results
            final_rmse = np.sqrt(-random_search.best_score_)
            self.logger.info(f"Best RMSE score: {final_rmse:.4f}")
            self.logger.info(f"Best parameters: {random_search.best_params_}")
            
            return random_search, X_test, y_test, self.rmse_callback.training_rmse
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise

    def evaluate_model(self, model: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data with proper winsorization"""
        try:
            # Apply winsorization to test features using limits learned from training
            X_test_winsorized = self.feature_winsorizer_.transform(X_test)
            y_pred = model.predict(X_test_winsorized)
            
            # For fair evaluation, also winsorize test targets using limits learned from training
            y_test_winsorized = self.target_winsorizer_.transform(
                y_test.values.reshape(-1, 1)
            ).ravel()
            
            # Calculate metrics using winsorized values
            rmse = np.sqrt(mean_squared_error(y_test_winsorized, y_pred))
            mae = np.mean(np.abs(y_test_winsorized - y_pred))
            mape = np.mean(np.abs((y_test_winsorized - y_pred) / y_test_winsorized)) * 100
            r2 = r2_score(y_test_winsorized, y_pred)
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
            
            self.logger.info("Model Evaluation Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric.upper()}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data using the trained model"""
        try:
            # Apply feature winsorization using limits learned during training
            X_winsorized = self.feature_winsorizer_.transform(X)
            return self.model.predict(X_winsorized)
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise