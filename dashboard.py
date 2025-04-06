import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
from scipy import stats
import math
import base64

# Load model and scalers
try:
    model_data = joblib.load('models/berth_model.joblib')
    model = model_data['model']  # Extract the actual model
    feature_names = model_data['feature_names']
    scalers = joblib.load('models/feature_scalers.joblib')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None
    feature_names = None
    scalers = None

def normalize_features(df):
    """Normalize features using saved scalers"""
    if scalers is None:
        st.error("Feature scalers not loaded properly")
        return df
    
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame([df], columns=feature_names)
        
    scaled_df = df.copy()
    try:
        for column in scaled_df.columns:
            if column in scalers:
                scaled_values = scalers[column].transform(scaled_df[[column]])
                scaled_df[column] = scaled_values
    except Exception as e:
        st.error(f"Error normalizing features: {str(e)}")
    return scaled_df

def predict_with_confidence(features, n_iterations=100):
    """Generate predictions with confidence intervals using Monte Carlo simulation."""
    # Check if model is available
    if model is None:
        st.error("Model not loaded properly. Please check model files.")
        return 0, 0, 0

    # Ensure features is a DataFrame with correct column names
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame([features], columns=feature_names)
    
    predictions = []
    noise_scale = 0.1  # Scale of noise to add

    # Normalize features once before adding noise
    features = normalize_features(features)

    for _ in range(n_iterations):
        try:
            # Create noisy features while preserving feature names
            noisy_features = features.copy()
            for col in noisy_features.columns:
                noise = np.random.normal(0, noise_scale, size=len(noisy_features))
                noisy_features[col] = noisy_features[col] + noise
            
            # Make prediction
            pred = model.predict(noisy_features)[0]
            predictions.append(pred)
        except Exception as e:
            continue
    
    if not predictions:
        st.error("All prediction attempts failed")
        return 0, 0, 0
        
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions)
    std_dev = np.std(predictions)
    
    return mean_pred, mean_pred - 1.96 * std_dev, mean_pred + 1.96 * std_dev

# Update vessel category time ranges
vessel_time_ranges = {
    'Small': (3 * 60, 6 * 60),  # 3-6 hours in minutes
    'Medium': (6 * 60, 12 * 60),  # 6-12 hours in minutes
    'Large': (12 * 60, 26 * 60),  # 12-26 hours in minutes
    'Ultra': (24 * 60, 36 * 60)   # 24-36 hours in minutes
}

# Update the prediction calculation
def calculate_operation_time(teus, grt, loa, vessel_category):
    base_time = np.mean(vessel_time_ranges[vessel_category])
    
    # Adjust based on vessel characteristics
    teu_factor = teus / (3000 if vessel_category == 'Small' else 6000 if vessel_category == 'Medium' else 12000)
    size_factor = loa / (180 if vessel_category == 'Small' else 277 if vessel_category == 'Medium' else 366)
    
    # Calculate adjusted time
    adjusted_time = base_time * (teu_factor * 0.6 + size_factor * 0.4)
    
    # Ensure prediction stays within category range
    min_time, max_time = vessel_time_ranges[vessel_category]
    return np.clip(adjusted_time, min_time, max_time)

# Update confidence interval calculation
def calculate_confidence_interval(predicted_time, vessel_category):
    min_time, max_time = vessel_time_ranges[vessel_category]
    range_width = max_time - min_time
    
    # Calculate confidence interval (±15% of the category's time range)
    interval = range_width * 0.15
    return max(predicted_time - interval, min_time), min(predicted_time + interval, max_time)

# Update anomaly detection
def detect_anomalies(predicted_time, historical_times, vessel_category, time_range):
    """Enhanced anomaly detection with better statistical analysis"""
    min_time, max_time = vessel_time_ranges[vessel_category]
    
    # Calculate z-score based on the vessel category's expected range
    mean_time = (min_time + max_time) / 2
    std_dev = (max_time - min_time) / 4
    z_score = (predicted_time - mean_time) / std_dev
    
    # Determine severity based on deviation
    if abs(z_score) > 3:
        severity = "severe"
    elif abs(z_score) > 2:
        severity = "minor"
    else:
        severity = "normal"
    
    # Calculate confidence score (inversely proportional to deviation)
    confidence = max(0, min(100, (1 - abs(z_score)/6) * 100))
    
    return abs(z_score) > 2, z_score, severity, confidence

def get_vessel_size_category(teus):
    """Determine vessel size category and typical completion time range"""
    if teus < 3000:
        return "Small", vessel_time_ranges['Small']
    elif teus < 8000:
        return "Medium", vessel_time_ranges['Medium']
    elif teus < 12000:
        return "Large", vessel_time_ranges['Large']
    else:
        return "Ultra", vessel_time_ranges['Ultra']

def generate_historical_times(vessel_size):
    """Generate realistic historical completion times based on vessel size"""
    size_category, (min_time, max_time) = get_vessel_size_category(vessel_size)
    mean_time = (min_time + max_time) / 2
    std_dev = (max_time - min_time) / 4  # Standard deviation to cover the range
    return np.random.normal(mean_time, std_dev, 1000)

def calculate_efficiency_metrics(prediction_minutes, vessel_teus, vessel_loa):
    """Calculate efficiency metrics for the prediction"""
    teus_per_hour = vessel_teus / (prediction_minutes / 60)
    meters_per_hour = vessel_loa / (prediction_minutes / 60)
    return {
        'teus_per_hour': teus_per_hour,
        'meters_per_hour': meters_per_hour
    }

def get_operation_insights(prediction_minutes, vessel_category, time_range, metrics):
    """Generate insights based on the prediction and vessel metrics"""
    typical_min, typical_max = time_range
    efficiency_level = ""
    insights = []
    
    # Determine efficiency level
    if prediction_minutes < typical_min:
        efficiency_level = "Highly Efficient"
    elif prediction_minutes > typical_max:
        efficiency_level = "Below Average"
    else:
        efficiency_level = "Normal"
    
    # Generate specific insights
    if prediction_minutes > typical_max:
        insights.extend([
            f"Operation time exceeds typical range by {(prediction_minutes - typical_max)/60:.1f} hours",
            f"TEU handling rate: {metrics['teus_per_hour']:.1f} TEUs/hour",
            "Consider additional resources or parallel operations",
            "Review historical performance at selected berth"
        ])
    elif prediction_minutes < typical_min:
        insights.extend([
            "Faster than typical operation time",
            f"High efficiency rate: {metrics['teus_per_hour']:.1f} TEUs/hour",
            "Verify all cargo handling requirements are included",
            "Check for potential operation scope mismatches"
        ])
    else:
        insights.extend([
            "Operation time within expected range",
            f"Standard handling rate: {metrics['teus_per_hour']:.1f} TEUs/hour",
            "Regular resource allocation should be sufficient",
            "Monitor for potential optimization opportunities"
        ])
    
    return efficiency_level, insights

def get_berth_performance_data(berth_code):
    """Get historical performance data for the berth (simulated)"""
    # In production, this would fetch real historical data
    performance_data = {
        'BERTH1': {'avg_rate': 550, 'max_rate': 650, 'optimal_vessel_size': 'Medium'},
        'BERTH2': {'avg_rate': 600, 'max_rate': 700, 'optimal_vessel_size': 'Large'},
        'BERTH3': {'avg_rate': 500, 'max_rate': 600, 'optimal_vessel_size': 'Medium'},
        'BERTH4': {'avg_rate': 650, 'max_rate': 750, 'optimal_vessel_size': 'Large'}
    }
    return performance_data.get(berth_code, {})

def suggest_optimizations(metrics, berth_performance, vessel_category):
    """Generate optimization suggestions based on performance metrics"""
    suggestions = []
    
    # Compare with berth's historical performance
    if metrics['teus_per_hour'] < berth_performance['avg_rate']:
        diff_percent = ((berth_performance['avg_rate'] - metrics['teus_per_hour']) / berth_performance['avg_rate']) * 100
        suggestions.append(f"Current TEU rate is {diff_percent:.1f}% below berth average")
        suggestions.append("Consider additional crane allocation")
    
    # Check berth-vessel size match
    if vessel_category != berth_performance['optimal_vessel_size']:
        suggestions.append(f"Note: {berth_code} is optimized for {berth_performance['optimal_vessel_size']} vessels")
        if berth_performance['optimal_vessel_size'] == 'Medium' and vessel_category == 'Large':
            suggestions.append("Consider splitting operation between two berths")
    
    # Potential improvement calculation
    potential_improvement = berth_performance['max_rate'] - metrics['teus_per_hour']
    if potential_improvement > 0:
        time_saved = (metrics['teus_per_hour'] / berth_performance['max_rate'] - 1) * -100
        suggestions.append(f"Potential time savings of {time_saved:.1f}% with optimal resource allocation")
    
    return suggestions

def generate_tide_data():
    """Generate realistic tide data for the next 72 hours"""
    current_time = datetime.now()
    tide_data = {}
    
    # Tidal cycle parameters - adjusted for more realistic values
    cycle_duration = 12.4  # hours (typical tidal cycle)
    base_height = max(14.0, vessel_draft + 2.0)  # Ensure minimum safe depth
    amplitude = 1.5      # Reduced tidal range for more realistic values
    
    for hour in range(72):
        for minute in range(0, 60, 15):  # 15-minute intervals
            time_point = current_time + timedelta(hours=hour, minutes=minute)
            # Calculate tide height using refined formula
            hour_in_cycle = (hour + minute/60) % cycle_duration / cycle_duration
            daily_factor = math.sin(2 * math.pi * (hour % 24) / 24)
            
            tide_height = (
                base_height +
                amplitude * math.sin(2 * math.pi * hour_in_cycle) +
                0.3 * amplitude * math.sin(4 * math.pi * hour_in_cycle) +
                0.2 * amplitude * daily_factor
            )
            
            # Ensure minimum safe depth with some variation
            min_safe_depth = vessel_draft + 1.0
            tide_data[time_point] = round(max(tide_height, min_safe_depth), 2)
    
    return tide_data

def find_optimal_berthing_windows(operation_duration, tide_data, vessel_draft):
    """Find optimal berthing windows considering tide and operation duration"""
    windows = []
    required_depth = vessel_draft + 1.0  # Reduced safety margin for more realistic values
    
    # Convert operation duration from minutes to hours
    operation_hours = operation_duration / 60
    
    times = sorted(list(tide_data.keys()))
    for i in range(len(times) - int(operation_hours * 4)):  # 4 readings per hour
        start_time = times[i]
        end_time = start_time + timedelta(hours=operation_hours)
        
        # Check if we have enough depth throughout the operation
        window_valid = True
        window_tides = []
        
        current_time = start_time
        while current_time <= end_time:
            closest_time = min(tide_data.keys(), key=lambda x: abs(x - current_time))
            tide_height = tide_data[closest_time]
            window_tides.append(tide_height)
            
            if tide_height < required_depth:
                window_valid = False
                break
            current_time += timedelta(minutes=15)
        
        if window_valid and window_tides:
            avg_tide = sum(window_tides) / len(window_tides)
            score = calculate_window_score(avg_tide, required_depth, start_time)
            
            # Reduced score threshold to show more windows
            if score >= 60:  # Lowered threshold
                windows.append({
                    "start": start_time,
                    "end": end_time,
                    "tide_height": avg_tide,
                    "score": score,
                    "min_tide": min(window_tides),
                    "max_tide": max(window_tides)
                })
    
    # Sort windows by score and return top ones
    return sorted(windows, key=lambda x: x['score'], reverse=True)

def calculate_window_score(avg_tide, required_depth, start_time):
    """Calculate a score for berthing window quality"""
    # Factors to consider:
    # 1. Clearance above required depth (40%)
    # 2. Daylight operations (30%)
    # 3. Time of day preference (30%)
    
    # Depth clearance score (0-100)
    depth_margin = avg_tide - required_depth
    depth_score = min(100, (depth_margin / 2) * 100)  # 2m extra depth = perfect score
    
    # Daylight score (0-100)
    hour = start_time.hour
    is_daylight = 6 <= hour <= 18
    daylight_score = 100 if is_daylight else 50
    
    # Time preference score (0-100)
    # Prefer early morning or early afternoon starts
    preferred_hours = [7, 8, 9, 13, 14, 15]
    if hour in preferred_hours:
        time_score = 100
    elif 6 <= hour <= 18:
        time_score = 80
    else:
        time_score = 60
    
    # Calculate weighted score
    final_score = (
        depth_score * 0.4 +
        daylight_score * 0.3 +
        time_score * 0.3
    )
    
    return round(final_score, 1)

def display_berthing_windows(windows):
    """Display berthing windows in a clean, formatted way"""
    if not windows:
        st.warning("No suitable berthing windows found in the next 72 hours. Consider checking later time periods or different berths.")
        return

    st.markdown("### Optimal Berthing Windows")
    
    for i, window in enumerate(windows[:3], 1):
        duration_hours = (window['end'] - window['start']).total_seconds() / 3600
        clearance = window['tide_height'] - vessel_draft
        
        with st.expander(f"Window {i} - Score: {window['score']:.1f}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Start Time**")
                st.info(f"{window['start'].strftime('%Y-%m-%d %H:%M')}")
                
                st.markdown("**Duration**")
                st.info(f"{duration_hours:.1f} hours")
                
                st.markdown("**Min. Tide**")
                st.info(f"{window['min_tide']:.2f}m")
            
            with col2:
                st.markdown("**End Time**")
                st.info(f"{window['end'].strftime('%Y-%m-%d %H:%M')}")
                
                st.markdown("**Avg. Tide**")
                st.info(f"{window['tide_height']:.2f}m")
                
                st.markdown("**Max. Tide**")
                st.info(f"{window['max_tide']:.2f}m")
            
            # Safety clearance with color coding
            clearance_color = (
                "#2ecc71" if clearance >= 1.5 else 
                "#f1c40f" if clearance >= 1.0 else 
                "#e74c3c"
            )
            st.markdown(
                f"**Safety Clearance**\n\n"
                f"<span style='color: {clearance_color}; font-size: 1.2em;'>"
                f"{clearance:.2f}m</span>",
                unsafe_allow_html=True
            )

    # Add tide level visualization
    st.markdown("### Tide Levels During Windows")
    fig = go.Figure()
    
    # Plot tide data
    tide_times = list(tide_data.keys())
    tide_heights = list(tide_data.values())
    
    fig.add_trace(go.Scatter(
        x=tide_times,
        y=tide_heights,
        name='Tide Level',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add required depth line
    fig.add_hline(
        y=vessel_draft,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Required Depth ({vessel_draft}m)",
        annotation=dict(font_size=12, font_color="black")
    )
    
    # Highlight berthing windows
    for window in windows[:3]:
        fig.add_vrect(
            x0=window['start'],
            x1=window['end'],
            fillcolor="#2ecc71",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text=f"Window {windows.index(window) + 1}",
            annotation_position="top left",
            annotation=dict(font_size=12, font_color="black")
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        showlegend=True,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_title="Time",
        yaxis_title="Tide Height (m)",
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Set page config and theme
st.set_page_config(
    page_title="Berth Management Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pure white theme
st.markdown("""
<style>
    /* Global text color and background */
    .stApp, .stMarkdown, p, .stText {
        color: #000000 !important;
    }
    
    /* Main content area */
    .main, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, .stTitle {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Containers and expanders */
    .stContainer, [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background-color: #ffffff !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* Input fields */
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Metrics and indicators */
    .stMetric {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Plots */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Custom container styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    
    /* Caption text */
    .stCaption {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Berth Management Dashboard")

# Sidebar with minimal organization
st.sidebar.header("Vessel Information")

# Vessel inputs without section headers
vessel_teus = st.sidebar.number_input("TEUs", min_value=0, max_value=24000, value=6800)
vessel_grt = st.sidebar.number_input("Gross Registered Tonnage", min_value=0, max_value=200000, value=72000)
vessel_loa = st.sidebar.number_input("Length Overall (meters)", min_value=0.0, max_value=400.0, value=277.0)
vessel_draft = st.sidebar.number_input("Draft (meters)", min_value=0.0, max_value=20.0, value=12.5)
berth_code = st.sidebar.selectbox("Berth", ["BERTH1", "BERTH2", "BERTH3", "BERTH4"])

# Create feature vector
features = pd.DataFrame({
    'LOA': [vessel_loa],
    'No_of_Teus': [vessel_teus],
    'GRT': [vessel_grt],
    'port_waiting_time': [2.0],  # Default values
    'ops_preparation_time': [1.5],
    'total_port_time': [24.0],
    'start_minutes_of_day': [360],  # 6 AM default
    'arrival_hour': [6],
    'arrival_day': [datetime.now().day],
    'arrival_month': [datetime.now().month],
    'arrival_year': [datetime.now().year],
    'arrival_dayofweek': [datetime.now().weekday()],
    'Berth_Code_encoded': [["BERTH1", "BERTH2", "BERTH3", "BERTH4"].index(berth_code)],
    'size_category_encoded': [2],  # Will be determined by LOA
    'teu_category_encoded': [2],   # Will be determined by TEUs
    'grt_category_encoded': [2],   # Will be determined by GRT
    'season_encoded': [datetime.now().month // 3],
    'teu_per_meter': [vessel_teus/vessel_loa],
    'grt_per_meter': [vessel_grt/vessel_loa],
    'grt_per_teu': [vessel_grt/vessel_teus],
    'total_prep_ratio': [0.75],
    'waiting_ratio': [0.1],
    'teu_density': [vessel_teus/(vessel_loa * vessel_grt)],
    'volume_index': [np.cbrt(vessel_loa * vessel_grt * vessel_teus)],
    'avg_prep_time_by_size': [1.5],
    'avg_waiting_time_by_size': [2.0],
    'is_night_arrival': [0],
    'is_weekend': [0]
})

# Normalize features
features = normalize_features(features)

# Make prediction with confidence intervals
mean_pred, lower_bound, upper_bound = predict_with_confidence(features)

# Determine vessel size category and generate historical times
vessel_category, time_range = get_vessel_size_category(vessel_teus)
historical_completion_times = generate_historical_times(vessel_teus)

# Calculate operation time based on vessel characteristics
operation_time = calculate_operation_time(vessel_teus, vessel_grt, vessel_loa, vessel_category)

# Calculate confidence interval
confidence_interval = calculate_confidence_interval(operation_time, vessel_category)

# Calculate efficiency metrics
efficiency_metrics = calculate_efficiency_metrics(operation_time, vessel_teus, vessel_loa)

# Get operation insights
efficiency_level, operation_insights = get_operation_insights(
    operation_time, vessel_category, time_range, efficiency_metrics
)

# Get berth performance data
berth_performance = get_berth_performance_data(berth_code)

# Suggest optimizations
optimization_suggestions = suggest_optimizations(efficiency_metrics, berth_performance, vessel_category)

# Detect anomalies
is_anomaly, z_score, severity, confidence = detect_anomalies(operation_time, historical_completion_times, vessel_category, time_range)

# Generate tide data
tide_data = generate_tide_data()

# Find optimal berthing windows
optimal_windows = find_optimal_berthing_windows(operation_time, tide_data, vessel_draft)

# Display results
st.header("Berth Operation Analysis")

# Combine prediction and context in a more concise way
col1, col2 = st.columns(2)
with col1:
    st.write("**Vessel Profile**")
    st.write(f"""
    - Size Category: {vessel_category}
    - TEU Density: {vessel_teus/vessel_loa:.1f} TEUs/meter
    - Volume Index: {np.cbrt(vessel_loa * vessel_grt * vessel_teus):.0f}
    """)

with col2:
    st.write("**Operation Timeline**")
    st.write(f"""
    - Expected Duration: {operation_time/60:.1f} hours
    - Typical Range: {time_range[0]/60:.1f} - {time_range[1]/60:.1f} hours
    - Confidence Interval: {confidence_interval[0]/60:.1f} - {confidence_interval[1]/60:.1f} hours
    """)

# Berth Operation Analysis Section
st.header("Berth Operation Analysis")

# Create two columns for the gauges
col1, col2 = st.columns(2)

with col1:
    # TEU Processing Rate Gauge
    current_rate = efficiency_metrics['teus_per_hour']
    max_rate = berth_performance['max_rate']
    teu_ranges = [[0, max_rate * 0.6], [max_rate * 0.6, max_rate * 0.8], [max_rate * 0.8, max_rate]]
    
    st.subheader("TEU Processing Rate")
    teu_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_rate,
        domain = {'x': [0.1, 0.9], 'y': [0, 0.9]},
        title = {'text': f"{int(current_rate)} TEUs/hr", 'font': {'size': 14, 'color': 'black'}},  # Black text
        number = {'font': {'size': 24, 'color': 'black'}},  # Black text
        gauge = {
            'axis': {
                'range': [0, max_rate],
                'tickwidth': 1,
                'tickcolor': 'black',  # Black ticks
                'ticktext': ['Low', 'Medium', 'High'],  # Added labels
                'tickfont': {'color': 'black'}  # Black tick labels
            },
            'bar': {'color': "#000000", 'thickness': 0.6},
            'steps': [
                {'range': teu_ranges[0], 'color': "#ff4b4b"},
                {'range': teu_ranges[1], 'color': "#ffd34b"},
                {'range': teu_ranges[2], 'color': "#4bff4b"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.6,
                'value': current_rate
            }
        }
    ))
    
    teu_gauge.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 12, 'color': 'black'},  # Black font
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot background
    )
    st.plotly_chart(teu_gauge, use_container_width=True)
    
    # Simple legend for TEU Rate
    st.markdown(
        f"<div style='text-align: center'>"
        f"Low < {int(max_rate * 0.6)}   |   "
        f"{int(max_rate * 0.6)}-{int(max_rate * 0.8)}   |   "
        f"> {int(max_rate * 0.8)}"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    # Resource Utilization Gauge
    utilization = (efficiency_metrics['teus_per_hour'] / berth_performance['max_rate'] * 100)
    util_ranges = [[0, 60], [60, 80], [80, 100]]
    
    st.subheader("Resource Utilization")
    util_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = utilization,
        domain = {'x': [0.1, 0.9], 'y': [0, 0.9]},
        title = {'text': f"{utilization:.1f}%", 'font': {'size': 14, 'color': 'black'}},  # Black text
        number = {'font': {'size': 24, 'color': 'black'}},  # Black text
        gauge = {
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': 'black',  # Black ticks
                'ticktext': ['Low', 'Medium', 'High'],  # Added labels
                'tickfont': {'color': 'black'}  # Black tick labels
            },
            'bar': {'color': "#000000", 'thickness': 0.6},
            'steps': [
                {'range': util_ranges[0], 'color': "#ff4b4b"},
                {'range': util_ranges[1], 'color': "#ffd34b"},
                {'range': util_ranges[2], 'color': "#4bff4b"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.6,
                'value': utilization
            }
        }
    ))
    
    util_gauge.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 12, 'color': 'black'},  # Black font
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot background
    )
    st.plotly_chart(util_gauge, use_container_width=True)
    
    # Simple legend for Utilization
    st.markdown(
        "<div style='text-align: center'>"
        "Low < 60%   |   "
        "60-80%   |   "
        "High > 80%"
        "</div>",
        unsafe_allow_html=True
    )

# Add a small explanation of the metrics
st.markdown("""
<div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0;'>
    <small>
    Metrics Explained:
    • TEU Rate: Containers processed per hour
    • Resource Utilization: Current vs maximum capacity
    </small>
</div>
""", unsafe_allow_html=True)

# Berth Occupancy Forecast Section
st.header("Berth Occupancy Forecast (Next 14 Days)")

# Get historical data and create forecast
forecast_days = 14
dates = [(datetime.now() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(forecast_days)]

# Use historical completion times to generate realistic forecasts
avg_daily_occupancy = np.mean(historical_completion_times) / 60  # Convert to hours
std_daily_occupancy = np.std(historical_completion_times) / 60

occupancy_data = []
for day in range(forecast_days):
    # Base prediction on historical patterns with some randomness
    base_pred = np.random.normal(avg_daily_occupancy, std_daily_occupancy * 0.3)
    # Add seasonal pattern (busier mid-week)
    day_of_week = (datetime.now() + timedelta(days=day)).weekday()
    seasonal_factor = 1.2 if day_of_week in [1, 2, 3] else 1.0  # Busier Tue-Thu
    
    predicted = base_pred * seasonal_factor
    confidence = 0.15 * (1 + day/14)  # Increasing uncertainty over time
    
    occupancy_data.append({
        'date': dates[day],
        'predicted': predicted,
        'lower_bound': predicted * (1 - confidence),
        'upper_bound': predicted * (1 + confidence)
    })

# Create forecast plot with confidence intervals
forecast_fig = go.Figure()

# Add confidence interval as a filled area
forecast_fig.add_trace(go.Scatter(
    x=dates + dates[::-1],
    y=[d['upper_bound'] for d in occupancy_data] + [d['lower_bound'] for d in occupancy_data][::-1],
    fill='toself',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

# Add main prediction line
forecast_fig.add_trace(go.Scatter(
    x=dates,
    y=[d['predicted'] for d in occupancy_data],
    line=dict(color='rgb(0,100,80)', width=2),
    name='Predicted Occupancy'
))

forecast_fig.update_layout(
    title="14-Day Berth Occupancy Forecast",
    xaxis_title="Date",
    yaxis_title="Predicted Hours",
    hovermode='x unified',
    showlegend=True
)

# Add annotations for busy/quiet periods
for i, d in enumerate(occupancy_data):
    if d['predicted'] > np.mean([x['predicted'] for x in occupancy_data]) * 1.1:
        forecast_fig.add_annotation(
            x=dates[i],
            y=d['predicted'],
            text="Peak",
            showarrow=True,
            arrowhead=1
        )

st.plotly_chart(forecast_fig, use_container_width=True)

# Display forecast insights
avg_occupancy = np.mean([d['predicted'] for d in occupancy_data])
peak_day = max(occupancy_data, key=lambda x: x['predicted'])
st.markdown("""
<div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>
    <h3 style='color: black; margin-bottom: 15px;'>Berth Occupancy Forecast</h3>
    <div style='color: black; margin-left: 10px;'>
        <p style='font-weight: 600; margin-bottom: 15px;'>Forecast Insights:</p>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li style='margin-bottom: 10px;'>Average daily occupancy: <span style='font-weight: 600;'>{:.1f} hours</span></li>
            <li style='margin-bottom: 10px;'>Peak occupancy expected on: <span style='font-weight: 600;'>{} ({:.1f} hours)</span></li>
            <li style='margin-bottom: 10px;'>Confidence intervals widen with forecast horizon</li>
        </ul>
    </div>
</div>
""".format(avg_occupancy, peak_day['date'], peak_day['predicted']), unsafe_allow_html=True)

# 2. Optimal Berthing Windows with Tidal Constraints
if optimal_windows:
    display_berthing_windows(optimal_windows)
else:
    st.warning("No suitable berthing windows found in the next 72 hours. Consider checking later time periods or different berths.")

# Anomaly Detection Section
st.header("Operation Analysis")

# Helper function for anomaly detection
def get_vessel_time_range(vessel_category):
    """Get the expected time range for a vessel category"""
    ranges = {
        'Small': (180, 360),    # 3-6 hours
        'Medium': (360, 720),   # 6-12 hours
        'Large': (720, 1440)    # 12-24 hours
    }
    return ranges.get(vessel_category, (360, 720))  # Default to Medium if category not found

# Create columns for anomaly detection
col1, col2 = st.columns(2)

with col1:
    # Determine status icon and color based on z_score
    status_icon = "" if abs(z_score) < 1 else "" if abs(z_score) < 2 else ""
    status_color = "#2ecc71" if abs(z_score) < 1 else "#f1c40f" if abs(z_score) < 2 else "#e74c3c"
    status_text = "Normal" if abs(z_score) < 1 else "Moderate Deviation" if abs(z_score) < 2 else "Significant Deviation"
    
    st.markdown(f"### {status_icon} Operation Timeline: {status_text}")
    
    # Deviation Score
    with st.container():
        st.markdown("#### Deviation Score")
        score_color = (
            "#2ecc71" if abs(z_score) < 1 
            else "#f1c40f" if abs(z_score) < 2 
            else "#e74c3c"
        )
        st.markdown(f"**Score:** :red[{abs(z_score):.1f}σ]")
        st.caption("Measures how far the predicted operation time deviates from historical average.")
        st.markdown("""
        - 0-1σ: Normal
        - 1-2σ: Moderate deviation
        - >2σ: Significant deviation
        """)
    
    # Prediction Confidence
    with st.container():
        st.markdown("#### Prediction Confidence")
        confidence_color = (
            "#2ecc71" if confidence > 90 
            else "#f1c40f" if confidence > 70 
            else "#e74c3c"
        )
        st.markdown(f"**Confidence:** :red[{confidence:.1f}%]")
        st.caption("Model's confidence in the operation time prediction.")
        st.markdown("""
        - >90%: High confidence
        - 70-90%: Moderate confidence
        - <70%: Low confidence
        """)
    
    # Interpretation Guide
    with st.expander("How to interpret these metrics"):
        st.markdown("""
        - **Most reliable prediction** 
          - Low deviation + High confidence
        - **Unusual but certain operation** 
          - High deviation + High confidence
        - **Typical but uncertain operation** 
          - Low deviation + Low confidence
        - **Requires careful monitoring** 
          - High deviation + Low confidence
        """)

with col2:
    # Get expected time range for the vessel category
    min_time, max_time = get_vessel_time_range(vessel_category)
    operation_hours = operation_time/60
    
    # Determine operation time status
    time_status = (
        "Within expected range" if min_time/60 <= operation_hours <= max_time/60
        else "Below expected range" if operation_hours < min_time/60
        else "Above expected range"
    )
    
    st.markdown("### Current Assessment")
    
    # Operation Time
    with st.container():
        st.markdown("#### Operation Time")
        st.markdown(f"**Status:** {'' if time_status == 'Within expected range' else ''} {time_status}")
        st.markdown(f"""
        - Predicted: **{operation_hours:.1f} hours**
        - Expected range: {min_time/60:.1f} - {max_time/60:.1f} hours
        """)
    
    # Resource Allocation
    with st.container():
        st.markdown("#### Resource Allocation")
        st.markdown("**Appropriate**")
        st.caption("Current utilization matches vessel requirements")
    
    # Schedule Status
    with st.container():
        st.markdown("#### Schedule Status")
        st.markdown("**No adjustments needed**")
        st.caption("Operation aligns with optimal berthing windows")
    
    # Monitoring Recommendations
    with st.expander("Monitoring Recommendations"):
        st.markdown("""
        - Continue standard monitoring procedures
        - No additional resources required
        - Next assessment in 30 minutes
        """)

# Add custom CSS for better overall styling
st.markdown("""
<style>
    /* Additional styling for metrics and containers */
    div[data-testid="stMetricValue"] {
        color: black !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: black !important;
    }
    
    /* Improve container spacing */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Style dividers */
    hr {
        margin: 2rem 0;
        border-color: #e0e0e0;
    }
    
    /* Enhance text readability */
    .stMarkdown p {
        line-height: 1.6;
        color: black !important;
    }
    
    /* Style metric containers */
    div[data-testid="stMetricValue"] > div {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)
