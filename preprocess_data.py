import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def add_feature_interactions(df):
    """Add meaningful feature interactions without using target variable"""
    # Vessel characteristics
    df['teu_per_meter'] = df['No_of_Teus'] / df['LOA']
    df['grt_per_meter'] = df['GRT'] / df['LOA']
    df['grt_per_teu'] = df['GRT'] / df['No_of_Teus']
    
    # Time-based features
    df['total_prep_ratio'] = df['ops_preparation_time'] / df['port_waiting_time']
    df['waiting_ratio'] = df['port_waiting_time'] / df['total_port_time']
    
    # Vessel density metrics
    df['teu_density'] = df['No_of_Teus'] / (df['LOA'] * df['GRT'])
    df['volume_index'] = np.cbrt(df['LOA'] * df['GRT'] * df['No_of_Teus'])
    
    # Time of day effects
    df['is_night_arrival'] = ((df['arrival_hour'] >= 20) | (df['arrival_hour'] <= 4)).astype(int)
    df['is_weekend'] = (df['arrival_dayofweek'] >= 5).astype(int)
    
    return df

def preprocess_data(input_file, output_file):
    """Preprocess the berth data for completion time prediction"""
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print("Converting date columns...")
    date_columns = ['Actual_Arrival', 'Arrival_at_Berth', 'Ops_Start_from', 
                   'Ops_Completed_On', 'DeParture_from_Berth', 'Dearture_from_Port']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='mixed')
    
    print("Creating time-based features...")
    # Calculate time differences
    df['total_port_time'] = (df['Dearture_from_Port'] - df['Actual_Arrival']).dt.total_seconds() / 3600
    df['port_waiting_time'] = (df['Arrival_at_Berth'] - df['Actual_Arrival']).dt.total_seconds() / 3600
    df['ops_preparation_time'] = (df['Ops_Start_from'] - df['Arrival_at_Berth']).dt.total_seconds() / 3600
    
    # Extract temporal features
    df['start_hour'] = df['Ops_Start_from'].dt.hour
    df['start_minute'] = df['Ops_Start_from'].dt.minute
    df['arrival_hour'] = df['Arrival_at_Berth'].dt.hour
    df['arrival_day'] = df['Arrival_at_Berth'].dt.day
    df['arrival_month'] = df['Arrival_at_Berth'].dt.month
    df['arrival_year'] = df['Arrival_at_Berth'].dt.year
    df['arrival_dayofweek'] = df['Arrival_at_Berth'].dt.dayofweek
    
    # Calculate minutes since start of day for better time representation
    df['start_minutes_of_day'] = df['start_hour'] * 60 + df['start_minute']
    
    print("Creating vessel categories...")
    # Create vessel categories
    df['size_category'] = pd.qcut(df['LOA'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    df['teu_category'] = pd.qcut(df['No_of_Teus'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['grt_category'] = pd.qcut(df['GRT'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Create seasonal categories
    df['season'] = pd.cut(df['arrival_month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    print("Encoding categorical variables...")
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Berth_Code', 'size_category', 'teu_category', 'grt_category', 'season']
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col])
    
    # Add feature interactions
    df = add_feature_interactions(df)
    
    # Historical averages by vessel size
    size_groups = df.groupby('size_category')
    df['avg_prep_time_by_size'] = df['size_category'].map(size_groups['ops_preparation_time'].mean())
    df['avg_waiting_time_by_size'] = df['size_category'].map(size_groups['port_waiting_time'].mean())
    
    # Calculate target: minutes until completion
    df['completion_minutes'] = ((df['Ops_Completed_On'] - df['Ops_Start_from']).dt.total_seconds() / 60).astype(int)
    
    # Replace infinities with NaN and fill NaN with median
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    print("Selecting final features...")
    features = [
        # Vessel characteristics
        'LOA', 'No_of_Teus', 'GRT',
        
        # Time-based features
        'port_waiting_time', 'ops_preparation_time', 'total_port_time',
        'start_minutes_of_day', 'arrival_hour', 'arrival_day',
        'arrival_month', 'arrival_year', 'arrival_dayofweek',
        
        # Encoded categorical features
        'Berth_Code_encoded', 'size_category_encoded',
        'teu_category_encoded', 'grt_category_encoded',
        'season_encoded',
        
        # Vessel metrics
        'teu_per_meter', 'grt_per_meter', 'grt_per_teu',
        
        # Time ratios
        'total_prep_ratio', 'waiting_ratio',
        
        # Density and volume
        'teu_density', 'volume_index',
        
        # Historical averages
        'avg_prep_time_by_size', 'avg_waiting_time_by_size',
        
        # Binary flags
        'is_night_arrival', 'is_weekend',
        
        # Target variable
        'completion_minutes'
    ]
    
    # Save processed data
    print(f"Saving processed data to {output_file}...")
    processed_df = df[features].copy()
    processed_df.to_csv(output_file, index=False)
    print("Done!")
    
    return processed_df

if __name__ == '__main__':
    input_file = 'Data_Berth.csv'
    output_file = 'processed_berth_data.csv'
    processed_df = preprocess_data(input_file, output_file)
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(processed_df)}")
    print(f"Number of features: {len(processed_df.columns)}")
    print("\nFeature names:")
    for col in processed_df.columns:
        print(f"- {col}")
