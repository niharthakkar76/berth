# Berth Management System Documentation

## Simple Overview: What Does This System Do?

Imagine you're running a busy seaport, like managing a parking lot for massive ships. Each ship carries thousands of containers, and you need to know exactly when they'll arrive, how long they'll stay, and where to park them. That's what our system helps with!

Here's what it does in simple terms:

Think of it like a smart assistant for the port. When a ship is coming, the system looks at information like:
- How big the ship is
- How many containers it's carrying
- Which parking spot (berth) would be best
- How long similar ships usually take to unload

Then, just like your phone's GPS tells you how long a car journey will take, our system tells the port:
- Exactly how many hours the ship will need at the berth
- The best time for the ship to arrive
- How many cranes and workers might be needed
- If there might be any delays or problems

The system uses historical data and pre-trained models to make predictions. While it doesn't automatically learn from new experiences, it can be periodically retrained by port operators with new data to improve its accuracy. This is like updating your GPS system with new maps - it needs manual updates to stay current.

Think of it as your port's personal assistant that:
1. Uses historical data to predict ship handling times
2. Finds the best parking spot based on current conditions
3. Tells you how long the work will take
4. Helps monitor operations in real-time
5. Suggests ways to work faster based on pre-defined rules

## Overview
An intelligent system for optimizing port operations through machine learning-based berth allocation and prediction.

## Components

### 1. Data Processing (`preprocess_data.py`)
- Cleans and validates vessel data
- Engineers features for ML model
- Handles temporal data processing
- Performs categorical encoding

### 2. Model Training (`model_training.py`)
- Implements LightGBM ensemble model
- Performs cross-validation
- Optimizes hyperparameters
- Saves trained models

### 3. Model Retraining (`retrain_model.py`)
- Handles incremental learning
- Monitors model performance
- Updates model with new data

### 4. Dashboard (`dashboard.py`)
- Provides interactive Streamlit interface
- Generates real-time predictions
- Displays interactive visualizations
- Offers optimization suggestions

## Key Features

### Prediction System
- Real-time berth occupancy forecasting
- Confidence interval estimation
- Bootstrap sampling (100 iterations)
- Uncertainty quantification

### Vessel Categories
- Small: <3,000 TEUs (3-6 hours)
- Medium: 3,000-8,000 TEUs (6-12 hours)
- Large: >8,000 TEUs (12-26 hours)

### Performance Metrics
- TEU handling rate
- Berth utilization
- Resource efficiency
- Schedule adherence

### Optimization
- Tidal window optimization
- Resource allocation
- Schedule optimization
- Anomaly detection

## Model Architecture and Training Details

### 1. Machine Learning Model
#### Core Model: LightGBM Ensemble
- **Architecture**: Gradient boosting framework
- **Base Learners**: Decision trees
- **Optimization**: Leaf-wise tree growth
- **Key Advantages**:
  - Handles categorical features natively
  - Efficient with large datasets
  - Fast training and inference
  - Good handling of missing values

#### Model Parameters
- Learning Rate: 0.01
- Number of Trees: 1000
- Max Depth: 8
- Min Data in Leaf: 20
- Feature Fraction: 0.8
- Bagging Fraction: 0.8
- Early Stopping Rounds: 50

### 2. Feature Engineering
#### Temporal Features
- Hour of day
- Day of week
- Month
- Season
- Holiday indicators
- Peak/Off-peak periods

#### Vessel Features
- Size category encoding
- TEU density (TEUs/meter)
- Historical performance metrics
- Vessel type indicators
- Previous visit statistics

#### Operational Features
- Recent berth utilization
- Port congestion metrics
- Equipment availability
- Tidal windows

### 3. Model Training Process
#### Data Preprocessing
1. **Missing Value Handling**
   - Numerical: Median imputation
   - Categorical: Mode imputation
   - Time-based: Forward fill

2. **Feature Scaling**
   - Numerical features: Standard scaling
   - Time-based features: Min-max scaling
   - Categorical features: Label encoding

3. **Feature Selection**
   - Correlation analysis
   - Feature importance ranking
   - Domain expert validation

#### Training Pipeline
1. **Data Split**
   - Training: 70%
   - Validation: 15%
   - Test: 15%
   - Time-based splitting to prevent data leakage

2. **Cross-Validation**
   - 5-fold time-series cross-validation
   - Rolling window validation
   - Performance metric: RMSE, MAE

3. **Hyperparameter Optimization**
   - Bayesian optimization
   - Grid search for key parameters
   - Cross-validation for each parameter set

### 4. Model Evaluation
#### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

#### Validation Checks
- Historical data consistency
- Seasonal pattern detection
- Anomaly identification
- Edge case handling

### 5. Prediction System
#### Real-time Predictions
- Bootstrap sampling (100 iterations)
- Confidence interval calculation
- Uncertainty quantification
- Anomaly detection

#### Adjustment Factors
- Weather conditions
- Port congestion
- Equipment availability
- Tidal constraints

### 6. Model Maintenance
#### Regular Updates
- Weekly retraining schedule
- Performance monitoring
- Data quality checks
- Drift detection

#### Version Control
- Model versioning
- Feature set tracking
- Performance benchmarking
- Rollback capability

## How the System Actually Works

The system uses a pre-trained LightGBM model that makes predictions based on historical data. Here's how it really works:

1. **Fixed Model, Not Learning**
   The system doesn't automatically learn or improve itself. Instead, it uses a model that was trained on historical data about ships, their sizes, and how long they took to handle.

2. **Manual Retraining Process**
   To keep the system up-to-date, port operators need to:
   - Collect new data about ship operations
   - Run the `retrain_model.py` script
   - This creates a new model with the latest data
   - The new model replaces the old one

3. **What the Model Actually Predicts**
   - Uses 13 specific measurements about each ship
   - Considers things like ship length, container count, and past performance
   - Makes predictions using patterns found in historical data
   - Doesn't adjust predictions based on current operations

4. **Accuracy Checks**
   The system measures its accuracy by checking:
   - How many predictions are within 15 minutes of actual time
   - How many are within 30 minutes
   - How many are within 60 minutes
   - The average error in minutes

Think of it like a calculator rather than an AI assistant - it's very good at processing numbers and finding patterns, but it needs humans to update it with new information.

## Prediction Case Study

### Example Vessel Analysis

#### 1. Input Parameters
```
Vessel Specifications:
- TEUs: 14,500 (Ultra Large Container Ship)
- GRT: 156,000 tons
- LOA: 366 meters
- Draft: 15.20 meters
- Assigned: BERTH2
```

#### 2. Vessel Profile Analysis
```
Classification Metrics:
- Size Category: Large (>8,000 TEUs)
- TEU Density: 39.6 TEUs/meter
  * Calculation: 14,500 TEUs ÷ 366 meters
  * Indicates high-density vessel configuration
- Volume Index: 9,390
  * Derived from: ∛(LOA × GRT × TEUs)
  * Suggests significant operational complexity
```

#### 3. Operational Predictions
```
Time Estimates:
- Expected Duration: 25.9 hours
- Operational Window: 12.0 - 26.0 hours
- Confidence Interval: 25.9 - 25.9 hours (high certainty)

Performance Analysis:
- TEU Handling Rate: 559.3 TEUs/hour
  * Calculation: 14,500 TEUs ÷ 25.9 hours
  * Represents 79.9% of maximum capacity (700 TEUs/hour)
- Berth Utilization: 14.1 meters/hour
  * Calculation: 366 meters ÷ 25.9 hours
```

#### 4. Efficiency Metrics
```
Berth Performance:
- Current Rate: 559.3 TEUs/hour
- Historical Average: 600 TEUs/hour
- Maximum Capacity: 700 TEUs/hour
- Utilization Rate: 79.9%
  * Calculation: (559.3 ÷ 700) × 100
```

#### 5. Operational Assessment
```
Status: Operation Timeline Optimal

Justification:
1. Duration within expected range for vessel size
2. Resource allocation matches vessel requirements
3. No immediate schedule adjustments needed
```

#### 6. Performance Insights
```
Efficiency Analysis:
1. Current Performance:
   - Operating at 79.9% of berth capacity
   - 6.8% below historical average
   - Acceptable for vessel size and complexity

2. Resource Utilization:
   - Standard resource allocation sufficient
   - Additional optimization possible
   - Potential for 20.1% efficiency improvement
```

#### 7. Optimization Recommendations
```
Short-term Improvements:
1. Crane Allocation:
   - Consider deploying additional cranes
   - Target: Reduce gap to historical average

2. Resource Optimization:
   - Current: 559.3 TEUs/hour
   - Target: 600 TEUs/hour (minimum)
   - Potential improvement: 40.7 TEUs/hour

3. Time Optimization:
   - Potential savings: 20.1% with optimal resources
   - Target duration: ~20.7 hours (theoretical)
```

#### 8. Risk Assessment
```
Operational Considerations:
1. Vessel Characteristics:
   - Ultra Large Container Ship
   - High TEU density (39.6 TEUs/meter)
   - Deep draft (15.20 meters)

2. Berth Compatibility:
   - BERTH2 suitable for large vessels
   - Draft requirements within safety margins
   - Adequate handling capacity
```

This case study demonstrates the system's capability to:
1. Process complex vessel parameters
2. Generate accurate time predictions
3. Provide detailed efficiency metrics
4. Identify optimization opportunities
5. Ensure safe and efficient operations

## Installation
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Usage
1. Enter vessel details:
   - TEUs
   - Length (LOA)
   - GRT
   - Draft
   - Berth Code
2. View predictions and insights
3. Check optimization suggestions

## Data Fields
- VCN: Vessel Call Number (e.g., MEL1000001)
- IMO: Vessel identifier
- LOA: Length Overall (meters)
- No_of_Teus: Container capacity
- GRT: Gross Registered Tonnage
- Berth_Code: Berth identifier
- Timestamps: Arrival, Operations, Departure
- Performance: Turn Around, Occupancy, Utilization

## Expected Outcomes
- 7-14 day berth occupancy forecasts
- Optimal berthing windows with tidal constraints
- Anomaly detection for utilization patterns
- Real-time prediction adjustments
- Optimization recommendations

"everytime a vessels calls a port unique number assigned to teach call" → VCN column (e.g., MEL1000001)
"unique vessel number" → IMO column (e.g., 1925590)
"name of the vessel(changes sometimes)" → Vessel_Name column (e.g., Marvin Block)
"length of a vessel(in m)" → LOA column (Length Overall in meters)
"this for which port this data is as of now this for melbourne" → Port_Code column (AUMEL = Melbourne)
"it's a dock when a vessel comes from the sea they are attached to a berth" → Berth_Code column (BRT001-BRT010)
"20ft is length of container equnit container, 2870 tues means these many containers would be there" → No_of_Teus column (TEU = Twenty-foot Equivalent Units)
"gross registered turange(weight of the vessel with cargo)" → GRT column
"when the vessel reached the port" → Actual_Arrival column
"when the vessel is pushed or pull to the berth" → Arrival_at_Berth column
"vessel was attached to the dock" → Ops_Start_from column
"when did we finish the offloading" → Ops_Completed_On column
"After loading and offloading is done" → DeParture_from_Berth column
"Unnamed: 13" → Dearture_from_Port column
"Vessel Turn this is n-I column" → Vessel Turn Around (time from arrival to departure)
"m-j" → Berth Occupency (time vessel occupied the berth)
"l-k" → Berth Utilization (actual operational time)
"Unnamed: 17" → Month column
For example, looking at the first entry:

Vessel MEL1000001 (VCN)
IMO number 1925590
Named "Marvin Block"
98 meters long
Arrived at Melbourne (AUMEL)
Used berth BRT001
Carried 2870 containers (TEUs)
Weighed 98,790 GRT
Total turn around time was about 15.45 hours
Berth occupancy was 14.76 hours
Actual operational time (utilization) was 14.35 hours
Feedback submitted

VCN - Vessel Call Number, a unique identifier for each vessel visit (format: MEL followed by 7 digits)
IMO - International Maritime Organization number, a unique identifier for the vessel
Vessel_Name - Name of the vessel
LOA - Length Overall of the vessel (in meters)
Port_Code - Port code (AUMEL represents Melbourne, Australia)
Berth_Code - Specific berth identifier (BRT001-BRT010)
No_of_Teus - Number of TEUs (Twenty-foot Equivalent Units) carried by the vessel
GRT - Gross Registered Tonnage of the vessel
Actual_Arrival - Timestamp when vessel arrived at port
Arrival_at_Berth - Timestamp when vessel arrived at its assigned berth
Ops_Start_from - Timestamp when operations started
Ops_Completed_On - Timestamp when operations completed
DeParture_from_Berth - Timestamp when vessel left the berth
Dearture_from_Port - Timestamp when vessel departed from port
Vessel Turn Around - Total time (in hours) vessel spent in port
Berth Occupency - Time (in hours) vessel occupied the berth
Berth Utilization - Actual operational time (in hours)
Month - Timestamp of the vessel's arrival at berth (appears to be used for monthly tracking)

we need this Expected Outcomes:
• Accurate berth occupancy forecasts (7-14 days ahead) with confidence intervals
• Identification of optimal berthing windows accounting for tidal constraints
• Anomaly detection system for unexpected berth utilization patterns
• Real-time adjustments to predictions based on vessel location and speed updates
• Optimization recommendations for berth allocation to maximize utilization
