import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import joblib

class BerthModel:
    def __init__(self, params=None):
        """Initialize the berth prediction model"""
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 8
        }
        
        self.params = params if params is not None else default_params
        self.model = LGBMRegressor(**self.params)
        self.feature_names = None
    
    def train_with_cv(self, X, y, n_splits=5):
        """Train model with cross-validation and return feature importance"""
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        importances = []
        scores_rmse = []
        scores_mae = []
        
        print("\nCross-validation Scores:")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric=['rmse', 'mae'],
                early_stopping_rounds=100,
                verbose=100
            )
            
            # Get predictions
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mae = np.mean(np.abs(y_val - y_pred))
            
            scores_rmse.append(rmse)
            scores_mae.append(mae)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            })
            importances.append(importance)
            
            print(f"\nFold {fold}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
        
        # Average feature importance across folds
        importance_df = pd.concat(importances)
        importance_df = importance_df.groupby('feature')['importance'].mean().reset_index()
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print("\nCross-validation Summary:")
        print(f"RMSE: {np.mean(scores_rmse):.4f} (+/- {np.std(scores_rmse):.4f})")
        print(f"MAE: {np.mean(scores_mae):.4f} (+/- {np.std(scores_mae):.4f})")
        
        # Train final model on all data
        print("\nTraining final model on all data...")
        self.model.fit(X, y, verbose=100)
        
        return self.model, importance_df
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        
        print("\nModel Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return {'rmse': rmse, 'mae': mae}
    
    def save_model(self):
        """Save the trained model and feature names"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params
        }
        joblib.dump(model_data, 'models/berth_model.joblib')
