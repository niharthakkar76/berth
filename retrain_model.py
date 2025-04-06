import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from model_training import BerthModel
import joblib
from datetime import timedelta

def normalize_features(df):
    """Normalize features using robust scaling to handle outliers"""
    # Features that need scaling
    scale_features = [
        'LOA', 'No_of_Teus', 'GRT', 'port_waiting_time',
        'ops_preparation_time', 'total_port_time', 'teu_per_meter',
        'grt_per_meter', 'grt_per_teu', 'teu_density', 'volume_index',
        'avg_prep_time_by_size', 'avg_waiting_time_by_size'
    ]
    
    # Initialize scalers
    scalers = {}
    scaled_df = df.copy()
    
    # Apply RobustScaler to handle outliers better
    for feature in scale_features:
        if feature in scaled_df:
            scalers[feature] = RobustScaler()
            scaled_df[feature] = scalers[feature].fit_transform(scaled_df[[feature]])
    
    # Save scalers for future use
    joblib.dump(scalers, 'models/feature_scalers.joblib')
    
    return scaled_df

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions with multiple metrics"""
    # Calculate basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate percentage of predictions within time windows
    within_15min = np.mean(np.abs(y_true - y_pred) <= 15) * 100
    within_30min = np.mean(np.abs(y_true - y_pred) <= 30) * 100
    within_60min = np.mean(np.abs(y_true - y_pred) <= 60) * 100
    
    print("\nPrediction Accuracy:")
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"Root Mean Square Error: {rmse:.2f} minutes")
    print(f"\nPredictions within time windows:")
    print(f"Within 15 minutes: {within_15min:.1f}%")
    print(f"Within 30 minutes: {within_30min:.1f}%")
    print(f"Within 60 minutes: {within_60min:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'within_15min': within_15min,
        'within_30min': within_30min,
        'within_60min': within_60min
    }

def main():
    # Load preprocessed data
    print("Loading data...")
    data = pd.read_csv('processed_berth_data.csv')
    
    # Normalize features
    print("Normalizing features...")
    data = normalize_features(data)
    
    # Prepare features and target
    features = [col for col in data.columns if col != 'completion_minutes']
    target = 'completion_minutes'
    
    # Split data with stratification based on completion time bins
    print("Splitting data...")
    data['completion_bins'] = pd.qcut(data[target], q=5, labels=False)
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=data['completion_bins']
    )
    
    # Initialize model with adjusted parameters for regression
    print("Training model...")
    model = BerthModel(
        params={
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
    )
    
    # Train with cross-validation
    model.model, importance = model.train_with_cv(X, y, n_splits=5)
    model.feature_names = features
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    importance = importance.sort_values('importance', ascending=False)
    print(importance.head(10))
    
    print("\nDone!")

if __name__ == "__main__":
    main()
