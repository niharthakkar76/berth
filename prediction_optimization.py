import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def detect_berth_anomalies(data, model, threshold=2.0):
    """
    Detect anomalies in berth utilization patterns.
    
    Args:
        data: DataFrame containing recent berth data
        model: Trained model for making predictions
        threshold: Number of standard deviations for anomaly detection
    
    Returns:
        DataFrame containing detected anomalies
    """
    # Make predictions on recent data
    features = data.drop(['Berth Utilization'], axis=1)
    predictions = model.predict(features)
    
    # Calculate residuals
    residuals = data['Berth Utilization'] - predictions
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Detect anomalies
    z_scores = (residuals - mean_residual) / std_residual
    anomaly_mask = np.abs(z_scores) > threshold
    
    # Create anomalies DataFrame
    anomalies = pd.DataFrame({
        'timestamp': pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=7),
            periods=len(data),
            freq='H'
        )[anomaly_mask],
        'actual_utilization': data['Berth Utilization'][anomaly_mask],
        'predicted_utilization': predictions[anomaly_mask],
        'z_score': z_scores[anomaly_mask]
    })
    
    return anomalies.sort_values('z_score', ascending=False)

def find_optimal_berthing_windows(forecast_df, min_window_hours=4, max_utilization=0.85):
    """
    Find optimal berthing windows based on forecasted utilization.
    
    Args:
        forecast_df: DataFrame with columns [datetime, predicted_utilization, lower_bound, upper_bound]
        min_window_hours: Minimum duration for a berthing window
        max_utilization: Maximum acceptable utilization threshold
    
    Returns:
        List of dictionaries containing optimal berthing windows
    """
    optimal_windows = []
    current_window = None
    
    for i, row in forecast_df.iterrows():
        # Check if utilization is below threshold
        if row['predicted_utilization'] <= max_utilization:
            if current_window is None:
                current_window = {
                    'start_time': row['datetime'],
                    'start_index': i,
                    'utilization_values': [row['predicted_utilization']]
                }
            else:
                current_window['utilization_values'].append(row['predicted_utilization'])
        else:
            if current_window is not None:
                window_hours = (forecast_df.loc[i-1, 'datetime'] - 
                              current_window['start_time']).total_seconds() / 3600
                
                if window_hours >= min_window_hours:
                    optimal_windows.append({
                        'start_time': current_window['start_time'],
                        'end_time': forecast_df.loc[i-1, 'datetime'],
                        'duration_hours': window_hours,
                        'expected_utilization': np.mean(current_window['utilization_values'])
                    })
                current_window = None
    
    # Check last window
    if current_window is not None:
        window_hours = (forecast_df.iloc[-1]['datetime'] - 
                       current_window['start_time']).total_seconds() / 3600
        if window_hours >= min_window_hours:
            optimal_windows.append({
                'start_time': current_window['start_time'],
                'end_time': forecast_df.iloc[-1]['datetime'],
                'duration_hours': window_hours,
                'expected_utilization': np.mean(current_window['utilization_values'])
            })
    
    # Sort windows by expected utilization
    return sorted(optimal_windows, key=lambda x: x['expected_utilization'])

def optimize_berth_allocation(vessels_df, berths, forecast_df):
    """
    Optimize berth allocation based on predicted utilization and vessel requirements.
    
    Args:
        vessels_df: DataFrame containing vessel information
        berths: List of available berths
        forecast_df: DataFrame with utilization forecasts
    
    Returns:
        DataFrame with optimized berth assignments
    """
    assignments = []
    
    # Sort vessels by priority (you can modify this based on your requirements)
    sorted_vessels = vessels_df.sort_values(['No_of_Teus', 'LOA'], ascending=[False, False])
    
    # Track berth availability
    berth_schedule = {berth: [] for berth in berths}
    
    for _, vessel in sorted_vessels.iterrows():
        best_berth = None
        best_start_time = None
        min_waiting_time = float('inf')
        
        arrival_time = vessel['Arrival_at_Berth']
        estimated_duration = vessel['predicted_utilization']
        
        # Find best berth with minimum waiting time
        for berth in berths:
            # Get berth's current schedule
            schedule = berth_schedule[berth]
            
            if not schedule:
                # Berth is completely free
                waiting_time = 0
                possible_start = arrival_time
            else:
                # Find first available slot after last operation
                last_end = schedule[-1]['end']
                waiting_time = max(0, (last_end - arrival_time).total_seconds() / 3600)
                possible_start = last_end
            
            # Check if this is the best option so far
            if waiting_time < min_waiting_time:
                # Verify utilization during the proposed window
                window_mask = ((forecast_df['datetime'] >= possible_start) & 
                             (forecast_df['datetime'] <= possible_start + timedelta(hours=estimated_duration)))
                
                if not window_mask.any() or forecast_df.loc[window_mask, 'predicted_utilization'].max() <= 0.85:
                    min_waiting_time = waiting_time
                    best_berth = berth
                    best_start_time = possible_start
        
        if best_berth is not None:
            # Assign vessel to best berth
            end_time = best_start_time + timedelta(hours=estimated_duration)
            
            assignment = {
                'vessel_id': vessel['VCN'],
                'berth': best_berth,
                'start_time': best_start_time,
                'end_time': end_time,
                'waiting_time': min_waiting_time
            }
            
            assignments.append(assignment)
            berth_schedule[best_berth].append({
                'start': best_start_time,
                'end': end_time
            })
    
    return pd.DataFrame(assignments)

def adjust_predictions(forecast_df, vessel_updates):
    """
    Adjust predictions based on real-time vessel updates.
    
    Args:
        forecast_df: DataFrame with utilization forecasts
        vessel_updates: DataFrame containing vessel location and speed updates
    
    Returns:
        DataFrame with adjusted predictions
    """
    adjusted_df = forecast_df.copy()
    
    for _, vessel in vessel_updates.iterrows():
        # Calculate estimated arrival time based on distance and speed
        hours_to_arrival = vessel['distance_nm'] / vessel['speed_knots']
        estimated_arrival = pd.Timestamp.now() + timedelta(hours=hours_to_arrival)
        
        # Find relevant time window in forecast
        window_mask = (adjusted_df['datetime'] >= estimated_arrival) & \
                     (adjusted_df['datetime'] <= estimated_arrival + timedelta(hours=24))
        
        if window_mask.any():
            # Adjust predictions in the window
            adjustment_factor = 1.0 + (vessel['speed_knots'] - 12) / 12  # Baseline speed of 12 knots
            adjusted_df.loc[window_mask, 'predicted_utilization'] *= adjustment_factor
            adjusted_df.loc[window_mask, 'lower_bound'] *= adjustment_factor
            adjusted_df.loc[window_mask, 'upper_bound'] *= adjustment_factor
    
    return adjusted_df
