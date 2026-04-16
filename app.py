import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon=":cloud:",
    layout="wide"
)

# Title and description
st.title("Weather Prediction App")
st.markdown("""
    This application predicts whether a day will be **sunny** or **not sunny** 
    based on weather parameters such as precipitation, temperature, and wind speed.
""")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('weather_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

# Sidebar for input
st.sidebar.header("Input Weather Parameters")

st.sidebar.markdown("---")
st.sidebar.subheader("Enter Weather Data:")

# Input fields
precipitation = st.sidebar.number_input(
    "Precipitation (mm)",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.5,
    help="Amount of rainfall in millimeters"
)

temp_max = st.sidebar.number_input(
    "Maximum Temperature (C)",
    min_value=-5.0,
    max_value=40.0,
    value=20.0,
    step=0.5,
    help="Highest temperature of the day"
)

temp_min = st.sidebar.number_input(
    "Minimum Temperature (C)",
    min_value=-10.0,
    max_value=30.0,
    value=12.0,
    step=0.5,
    help="Lowest temperature of the day"
)

wind = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5,
    help="Average wind speed"
)

day = st.sidebar.slider(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15
)

month = st.sidebar.selectbox(
    "Month",
    options=list(range(1, 13)),
    format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1],
    index=5
)

year = st.sidebar.selectbox(
    "Year",
    options=[2012, 2013, 2014, 2015],
    index=2
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
    index=0
)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    input_data = {
        "Precipitation": f"{precipitation} mm",
        "Max Temperature": f"{temp_max} C",
        "Min Temperature": f"{temp_min} C",
        "Wind Speed": f"{wind} m/s",
        "Day": day,
        "Month": month,
        "Year": year,
        "Day of Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]
    }
    
    for key, value in input_data.items():
        st.metric(key, value)

with col2:
    st.subheader("Prediction Result")
    
    # Load model and make prediction
    model = load_model()
    
    if model is None:
        st.warning("Model not found. Please train the model first.")
        st.info("Run 'python train_model.py' to train and save the model.")
    else:
        features = [precipitation, temp_max, temp_min, wind, day, month, year, day_of_week]
        feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'day_of_week']
        input_df = pd.DataFrame([features], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.success("SUNNY")
            st.balloons()
        else:
            st.error("NOT SUNNY")
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_df)[0][1]
            st.progress(float(prob))
            st.caption(f"Sunny Probability: {prob*100:.1f}%")

# Visualization section
st.markdown("---")
st.subheader("Data Visualization")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('seattle-weather.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['is_sunny'] = (df['weather'] == 'sun').astype(int)
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    tab1, tab2, tab3 = st.tabs(["Weather Distribution", "Monthly Trends", "Correlation Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            weather_counts = df['weather'].value_counts()
            ax.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title("Distribution of Weather Types")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sunny_counts = df['is_sunny'].value_counts()
            ax.pie(sunny_counts.values, labels=['Not Sunny', 'Sunny'], autopct='%1.1f%%', startangle=90)
            ax.set_title("Sunny vs Not Sunny")
            st.pyplot(fig)
    
    with tab2:
        monthly_sunny = df.groupby('month')['is_sunny'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        bars = ax.bar(months, monthly_sunny.values, color='skyblue', edgecolor='navy')
        ax.set_xlabel("Month")
        ax.set_ylabel("Sunny Days (%)")
        ax.set_title("Percentage of Sunny Days by Month")
        ax.set_ylim(0, 100)
        
        for bar, val in zip(bars, monthly_sunny.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
    
    with tab3:
        numeric_cols = ['precipitation', 'temp_max', 'temp_min', 'wind', 'is_sunny']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

else:
    st.info("Data file not found. Please ensure 'seattle-weather.csv' is available.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Weather Prediction App | Built with Streamlit & Scikit-learn</p>
        <p>Model trained on Seattle weather data (2012-2015)</p>
    </div>
""", unsafe_allow_html=True)
