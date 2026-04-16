import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.markdown("""
    Aplikasi ini memprediksi apakah suatu hari akan **cerah (sunny)** atau **tidak cerah** berdasarkan 
    parameter cuaca seperti curah hujan, suhu maksimum, suhu minimum, dan kecepatan angin.
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
st.sidebar.header("📊 Input Parameter Cuaca")

st.sidebar.markdown("---")
st.sidebar.subheader("Masukkan Data Cuaca:")

# Input fields
precipitation = st.sidebar.number_input(
    "💧 Curah Hujan (mm)",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.5,
    help="Jumlah curah hujan dalam milimeter"
)

temp_max = st.sidebar.number_input(
    "🌡️ Suhu Maksimum (°C)",
    min_value=-5.0,
    max_value=40.0,
    value=20.0,
    step=0.5,
    help="Suhu tertinggi dalam sehari"
)

temp_min = st.sidebar.number_input(
    "🌡️ Suhu Minimum (°C)",
    min_value=-10.0,
    max_value=30.0,
    value=12.0,
    step=0.5,
    help="Suhu terendah dalam sehari"
)

wind = st.sidebar.number_input(
    "💨 Kecepatan Angin (m/s)",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5,
    help="Kecepatan angin rata-rata"
)

# Additional features from the notebook
day = st.sidebar.slider(
    "📅 Hari ke-",
    min_value=1,
    max_value=31,
    value=15,
    help="Tanggal dalam bulan"
)

month = st.sidebar.selectbox(
    "🗓️ Bulan",
    options=list(range(1, 13)),
    format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", 
                           "Jul", "Ags", "Sep", "Okt", "Nov", "Des"][x-1],
    index=5
)

year = st.sidebar.selectbox(
    "📆 Tahun",
    options=[2012, 2013, 2014, 2015],
    index=2
)

day_of_week = st.sidebar.selectbox(
    "📌 Hari dalam Minggu",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x],
    index=0
)

# Make prediction
def make_prediction(model, features):
    """Make prediction using loaded model"""
    if model is None:
        return None, None
    
    # Create DataFrame with feature names matching training data
    feature_names = ['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'day_of_week']
    input_df = pd.DataFrame([features], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
    
    return prediction, probability

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Ringkasan Input")
    input_data = {
        "Curah Hujan": f"{precipitation} mm",
        "Suhu Maksimum": f"{temp_max} °C",
        "Suhu Minimum": f"{temp_min} °C",
        "Kecepatan Angin": f"{wind} m/s",
        "Hari ke-": day,
        "Bulan": month,
        "Tahun": year,
        "Hari dalam Minggu": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][day_of_week]
    }
    
    for key, value in input_data.items():
        st.metric(key, value)

with col2:
    st.subheader("🎯 Hasil Prediksi")
    
    # Load model and make prediction
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model belum dilatih! Silakan latih model terlebih dahulu.")
        st.info("""
            **Cara melatih model:**
            1. Jalankan notebook yang tersedia
            2. Simpan model menggunakan pickle
            3. Pastikan file 'weather_model.pkl' berada di direktori yang sama
        """)
    else:
        features = [precipitation, temp_max, temp_min, wind, day, month, year, day_of_week]
        prediction, probability = make_prediction(model, features)
        
        if prediction == 1:
            st.success("### ☀️ **Cerah (Sunny)**")
            st.balloons()
        else:
            st.error("### 🌧️ **Tidak Cerah (Not Sunny)**")
        
        if probability is not None:
            st.progress(float(probability[1]))
            st.caption(f"Probabilitas Cerah: {probability[1]*100:.1f}%")

# Visualization section
st.markdown("---")
st.subheader("📈 Visualisasi Data Cuaca")

# Load original data for visualization
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
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Data", "📈 Tren Cuaca", "🔥 Korelasi", "📅 Pola Musiman"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of weather types
            weather_counts = df['weather'].value_counts()
            fig = px.pie(
                values=weather_counts.values,
                names=weather_counts.index,
                title="Distribusi Jenis Cuaca",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sunny vs Not Sunny
            sunny_counts = df['is_sunny'].value_counts()
            fig = px.pie(
                values=sunny_counts.values,
                names=['Tidak Cerah', 'Cerah'],
                title="Proporsi Hari Cerah vs Tidak Cerah",
                color_discrete_sequence=['#ff9999', '#66b3ff']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Time series of weather
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("Suhu Maksimum dan Minimum", "Curah Hujan dan Kecepatan Angin"))
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['temp_max'], name='Suhu Maksimum', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['temp_min'], name='Suhu Minimum', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['precipitation'], name='Curah Hujan', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['wind'], name='Kecepatan Angin', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Tren Cuaca dari Waktu ke Waktu")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        numeric_cols = ['precipitation', 'temp_max', 'temp_min', 'wind', 'day', 'month', 'year', 'day_of_week', 'is_sunny']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Matriks Korelasi Antar Fitur"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly sunny days
            monthly_sunny = df.groupby('month')['is_sunny'].mean() * 100
            fig = px.bar(
                x=monthly_sunny.index,
                y=monthly_sunny.values,
                title="Persentase Hari Cerah per Bulan",
                labels={'x': 'Bulan', 'y': 'Persentase Cerah (%)'},
                color=monthly_sunny.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
                                                 'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature by weather
            fig = px.box(
                df, x='weather', y='temp_max',
                title="Distribusi Suhu Maksimum per Jenis Cuaca",
                labels={'weather': 'Jenis Cuaca', 'temp_max': 'Suhu Maksimum (°C)'},
                color='weather'
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📁 Data tidak ditemukan. Pastikan file 'seattle-weather.csv' tersedia untuk visualisasi.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Aplikasi Prediksi Cuaca | Dibuat dengan Streamlit</p>
        <p>Model dilatih menggunakan data cuaca Seattle 2012-2015</p>
    </div>
""", unsafe_allow_html=True)
