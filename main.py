import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from model_training import BerthModel
from prediction_optimization import (
    detect_berth_anomalies,
    optimize_berth_allocation,
    find_optimal_berthing_windows
)
from datetime import datetime, timedelta

def preprocess_data():
    # Load preprocessed data
    data = pd.read_csv('processed_berth_data.csv')
    
    # Split features and target
    X = data.drop(['Berth Utilization'], axis=1)
    y = data['Berth Utilization']  # Predict actual operational time
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

def generate_future_features(X_test, days):
    # Generate 14-day forecast
    future_dates = pd.date_range(
        start=pd.Timestamp.now(),
        periods=days*24,  # days * 24 hours
        freq='H'
    )
    
    # Create features for future dates
    future_features = pd.DataFrame()
    future_features['arrival_hour'] = future_dates.hour
    future_features['arrival_day'] = future_dates.day
    future_features['arrival_month'] = future_dates.month
    future_features['arrival_year'] = future_dates.year
    future_features['arrival_dayofweek'] = future_dates.dayofweek
    
    # Use mean values from training data for other features
    for col in X_test.columns:
        if col not in future_features.columns:
            future_features[col] = X_test[col].mean()
    
    return future_features

def find_optimal_windows(predictions, lower_bounds, upper_bounds):
    # Find optimal berthing windows
    optimal_windows = []
    for i in range(len(predictions) - 4):  # Assuming 4-hour window
        window_avg = np.mean(predictions[i:i+4])
        if window_avg < 0.85:  # 85% maximum utilization threshold
            optimal_windows.append({
                'start': i,
                'end': i+4,
                'utilization': window_avg
            })
    
    return optimal_windows

def detect_anomalies(predictions, lower_bounds, upper_bounds):
    # Detect anomalies in recent data
    anomalies = []
    for i in range(len(predictions)):
        if predictions[i] > upper_bounds[i] or predictions[i] < lower_bounds[i]:
            anomalies.append({
                'time': i,
                'actual': predictions[i],
                'expected': np.mean(predictions),
                'zscore': (predictions[i] - np.mean(predictions)) / np.std(predictions)
            })
    
    return anomalies

def main():
    """Main function to run the berth prediction system"""
    print("Loading preprocessed data...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data()
    
    # Initialize and train model
    print("\nTraining model...")
    model = BerthModel()
    
    # Train with cross-validation
    model.model, feature_importance = model.train_with_cv(
        pd.concat([X_train, X_test]), 
        pd.concat([y_train, y_test])
    )
    
    # Store feature names
    model.feature_names = feature_cols
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print("-----------------")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Generate future predictions
    print("\nGenerating 14-day forecast...")
    future_features = generate_future_features(X_test, days=14)
    predictions, lower_bounds, upper_bounds = model.predict(future_features, return_confidence=True)
    
    # Find optimal berthing windows
    print("\nFinding optimal berthing windows...")
    optimal_windows = find_optimal_windows(predictions, lower_bounds, upper_bounds)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomalies = detect_anomalies(predictions, lower_bounds, upper_bounds)
    
    # Print summary
    print("\nForecast Summary:")
    print("-----------------")
    print(f"Average predicted utilization: {np.mean(predictions):.2f}")
    print(f"Peak utilization period: {future_features.index[np.argmax(predictions)]}")
    
    print("\nOptimal Berthing Windows:")
    print("------------------------")
    if optimal_windows:
        for window in optimal_windows:
            print(f"Start: {window['start']}")
            print(f"End: {window['end']}")
            print(f"Expected utilization: {window['utilization']:.2f}")
            print("---")
    else:
        print("No optimal berthing windows found")
    
    print("\nRecent Anomalies:")
    print("----------------")
    if anomalies:
        for anomaly in anomalies:
            print(f"Time: {anomaly['time']}")
            print(f"Actual vs Expected: {anomaly['actual']:.2f} vs {anomaly['expected']:.2f}")
            print(f"Z-score: {anomaly['zscore']:.2f}")
            print("---")
    else:
        print("No anomalies detected")
    
    # Save model
    print("\nSaving model...")
    model.save_model()

if __name__ == '__main__':
    main()
