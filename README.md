# Berth Prediction and Optimization System

This project implements an ensemble machine learning system for berth occupancy prediction and optimization at the Port of Melbourne.

## Features

- Accurate berth occupancy forecasts (7-14 days ahead) with confidence intervals
- Identification of optimal berthing windows considering tidal constraints
- Anomaly detection system for unexpected berth utilization patterns
- Real-time adjustments to predictions based on vessel location and speed
- Optimization recommendations for berth allocation

## Project Structure

- `data_preprocessing.py`: Data cleaning and feature engineering
- `model_training.py`: Ensemble model training and evaluation
- `prediction_optimization.py`: Prediction adjustments and berth optimization
- `main.py`: Main script to run the entire pipeline
- `requirements.txt`: Required Python packages

## Models Used

1. Random Forest Regressor
2. XGBoost Regressor
3. LightGBM Regressor
4. CatBoost Regressor
5. Prophet (for time series forecasting)

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Output

The system generates:
- Model performance metrics
- Feature importance visualizations
- Prediction vs actual plots
- Optimal berthing windows
- Anomaly detection results

## Data Features

- VCN: Vessel Call Number (unique identifier)
- IMO: International Maritime Organization number
- Vessel_Name: Name of the vessel
- LOA: Length Overall (meters)
- Port_Code: Port identifier
- Berth_Code: Berth identifier
- No_of_Teus: Number of container units
- GRT: Gross Registered Tonnage
- Various timestamp columns for vessel movements
- Performance metrics (Turn Around, Occupancy, Utilization)
