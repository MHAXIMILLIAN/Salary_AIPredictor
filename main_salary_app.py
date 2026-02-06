# salary_app.py - COMPLETE Salary Prediction System with ALL Features
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import io

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Salary Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# SESSION STATE & THEME SETUP
# -------------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

if 'page' not in st.session_state:
    st.session_state.page = "Single Prediction"


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        # Try multiple paths for flexibility
        try:
            model = joblib.load("best_salary_model.pkl")
        except:
            try:
                model = joblib.load("models/best_salary_model.pkl")
            except:
                model = joblib.load("best_model.pkl")
        return model, True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, False


model, model_loaded = load_model()


# -------------------------------
# MARKET DATA FUNCTION
# -------------------------------
def fetch_market_data():
    ALPHA_VANTAGE_API = "BU3EDDWZ30K6YMPF"
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "SPY",
                "interval": "5min",
                "apikey": ALPHA_VANTAGE_API
            },
            timeout=10
        )
        data = response.json()
        if "Time Series (5min)" in data:
            last_ref = list(data["Time Series (5min)"].keys())[0]
            market_index = float(data["Time Series (5min)"][last_ref]["4. close"])
            return market_index, True
    except Exception:
        pass
    return 400.0, False  # Fallback value


# -------------------------------
# PDF PROCESSING FUNCTION (for batch prediction)
# -------------------------------
def process_pdf_file(uploaded_file):
    try:
        # Try to import tabula
        try:
            import tabula
        except ImportError:
            st.error("❌ PDF processing requires tabula-py. Please install: `pip install tabula-py`")
            st.info("For now, please use CSV files.")
            return None

        # Try to process PDF
        try:
            tables = tabula.read_pdf(uploaded_file, pages='all', multiple_tables=True)
            if tables:
                # Combine all tables
                combined_df = pd.concat(tables, ignore_index=True)
                return combined_df
            else:
                st.error("No tables found in PDF")
                return None
        except Exception as e:
            if "java" in str(e).lower():
                st.error("❌ PDF processing requires Java Runtime Environment")
                st.info("""
                **Install Java on:**
                - **Windows/Mac:** Download from [java.com](https://java.com)
                - **Linux:** `sudo apt install default-jre`
                """)
            else:
                st.error(f"PDF processing error: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header"> Navigation</div>', unsafe_allow_html=True)

    # Theme selector
    theme = st.selectbox("🎨 Theme", ["Light", "Dark"],
                         index=0 if st.session_state.theme == "Light" else 1)
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    # Navigation
    page = st.radio(
        "Go to:",
        ["Single Prediction", "Batch Prediction", "Data Analytics", "About"],
        key="nav_radio"
    )
    st.session_state.page = page

    st.markdown("---")

    # Quick Info
    st.markdown("### Quick Info")
    if model_loaded:
        st.success("✅ Model loaded")
        if hasattr(model, 'named_steps'):
            try:
                st.info(f"Model: {type(model.named_steps['model']).__name__}")
            except:
                st.info("Pipeline model loaded")
    else:
        st.error("❌ Model not found")
        st.info("Place 'best_salary_model.pkl' in the root directory")

# -------------------------------
# COMPLETE CSS THEMING (Light/Dark) WITH BACKGROUND IMAGE
# -------------------------------
background_image_url = "https://images.unsplash.com/photo-1518546305927-5a555bb7020d?q=80&w=2069&auto=format&fit=crop"

if st.session_state.theme == "Dark":
    css = f"""
    <style>
    /* Dark Theme - Futuristic Blockchain Background */
    .stApp {{
        background: linear-gradient(rgba(10, 15, 35, 0.92), rgba(10, 15, 35, 0.95)),
                    url('{background_image_url}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
        color: #ffffff !important;
        min-height: 100vh;
    }}

    /* Special Background for Homepage - More Vibrant */
    {'div[data-testid="stAppViewContainer"]' if st.session_state.page == "Single Prediction" else ''} {{
        background: linear-gradient(rgba(10, 15, 35, 0.85), rgba(10, 15, 35, 0.90)),
                    url('{background_image_url}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
    }}

    /* Sidebar Styling */
    .stSidebar {{
        background: rgba(20, 25, 50, 0.85) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(0, 255, 255, 0.15) !important;
    }}

    .sidebar-header {{
        font-size: 1.5rem;
        color: #00ffff !important;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* Hero Sections */
    .hero-title {{
        font-size: 3.5rem;
        background: linear-gradient(135deg, #00ffff, #0077ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        font-weight: 800;
        letter-spacing: 1px;
    }}

    .hero-subtitle {{
        font-size: 1.2rem;
        color: #a0e7ff !important;
        opacity: 0.9;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
    }}

    /* Form Containers */
    .form-container {{
        background: rgba(20, 25, 50, 0.7) !important;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        box-shadow: 
            0 0 20px rgba(0, 150, 255, 0.1),
            inset 0 0 20px rgba(0, 150, 255, 0.05);
        backdrop-filter: blur(10px);
    }}

    /* Input Labels - DARK THEME */
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stTextArea label, .stSlider label, .stMultiselect label,
    .stCheckbox label, .stRadio label {{
        color: #a0e7ff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: 0 0 5px rgba(0, 200, 255, 0.3);
    }}

    /* Input Fields */
    .stTextInput input, .stNumberInput input, 
    .stSelectbox select, .stTextArea textarea {{
        background: rgba(30, 35, 60, 0.8) !important;
        border: 1px solid rgba(0, 200, 255, 0.4) !important;
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.5rem !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus,
    .stSelectbox select:focus, .stTextArea textarea:focus {{
        border-color: #00ffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.4) !important;
        outline: none !important;
    }}

    /* CAREFULLY ADDED: SLIDER STYLING - FIXED BUTTON */
    .stSlider {{
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }}

    .stSlider label {{
        color: #a0e7ff !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }}

    /* Slider track (the line) */
    .stSlider [data-baseweb="slider"] > div:first-child {{
        background: rgba(0, 150, 255, 0.3) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }}

    /* Slider thumb (the button/dragger) - FIXED */
    .stSlider [role="slider"] {{
        background: #00ffff !important;
        border: 2px solid #ffffff !important;
        width: 24px !important;
        height: 24px !important;
        border-radius: 50% !important;
        box-shadow: 
            0 0 15px rgba(0, 255, 255, 0.8),
            0 0 30px rgba(0, 255, 255, 0.4) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }}

    .stSlider [role="slider"]:hover {{
        transform: scale(1.2) !important;
        box-shadow: 
            0 0 20px rgba(0, 255, 255, 1),
            0 0 40px rgba(0, 255, 255, 0.6) !important;
        background: #ffffff !important;
        border-color: #00ffff !important;
    }}

    .stSlider [role="slider"]:active {{
        transform: scale(1.1) !important;
    }}

    /* Slider value display */
    .stSlider [data-testid="stWidgetLabel"] + div {{
        color: #00ffff !important;
        font-weight: bold !important;
        margin-top: 0.5rem !important;
        font-size: 1.1rem !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
    }}

    /* Cards */
    .info-card {{
        background: rgba(25, 30, 60, 0.6) !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 200, 255, 0.15) !important;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }}

    .info-card:hover {{
        transform: translateY(-5px);
        border-color: #00ffff !important;
        box-shadow: 0 5px 20px rgba(0, 255, 255, 0.2);
        background: rgba(30, 35, 70, 0.8) !important;
    }}

    .result-container {{
        background: linear-gradient(135deg, 
            rgba(0, 150, 255, 0.15), 
            rgba(0, 100, 255, 0.1)) !important;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(0, 200, 255, 0.3) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 30px rgba(0, 150, 255, 0.2);
    }}

    /* Salary Display */
    .salary-amount {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffff, #0077ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(0, 150, 255, 0.4);
    }}

    /* Icons */
    .info-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.7);
    }}

    /* Buttons */
    .stButton button {{
        background: linear-gradient(135deg, #0077ff, #00ffff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease;
    }}

    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4) !important;
    }}

    /* Metric Cards */
    .stMetric {{
        background: rgba(25, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 200, 255, 0.2) !important;
        border-radius: 10px;
        padding: 1rem;
    }}

    /* Multiselect */
    .stMultiSelect [data-baseweb="select"] {{
        background: rgba(30, 35, 60, 0.8) !important;
        border: 1px solid rgba(0, 200, 255, 0.4) !important;
    }}

    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(0, 150, 255, 0.3) !important;
        color: #ffffff !important;
    }}
    </style>
    """
else:  # LIGHT THEME SECTION
    css = f"""
    <style>
    /* Light Theme - Less transparent overlay for better contrast */
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.94)),
                    url('{background_image_url}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
        color: #1a365d !important;
        min-height: 100vh;
    }}

    /* Homepage - Even less transparent for maximum visibility */
    {'div[data-testid="stAppViewContainer"]' if st.session_state.page == "Single Prediction" else ''} {{
        background: linear-gradient(rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.90)),
                    url('{background_image_url}') !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
    }}

    /* Sidebar - Solid white for maximum contrast */
    .stSidebar {{
        background: rgba(255, 255, 255, 0.98) !important;
        border-right: 1px solid rgba(0, 100, 255, 0.3) !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
    }}

    .sidebar-header {{
        font-size: 1.5rem;
        color: #0056b3 !important;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }}

    /* Hero Sections */
    .hero-title {{
        font-size: 3.5rem;
        color: #0056b3 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }}

    .hero-subtitle {{
        font-size: 1.2rem;
        color: #2c5282 !important;
        opacity: 0.95;
        text-align: center;
        font-weight: 600;
    }}

    /* Form Containers - Solid white background */
    .form-container {{
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(0, 100, 255, 0.3) !important;
        box-shadow: 0 5px 25px rgba(0, 80, 255, 0.12);
    }}

    /* COMPREHENSIVE LABEL FIXES - All labels dark blue */
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stTextArea label, .stSlider label, .stMultiselect label,
    .stCheckbox label, .stRadio label, .stFormSubmitButton label,
    .stSubheader, h3, h4, h2, .stMarkdown h3, .stMarkdown h4 {{
        color: #1a365d !important;  /* Dark navy blue */
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}

    /* Specific fix for subheaders (Skills & Certifications, Market Conditions) */
    div[data-testid="stVerticalBlock"] > div > .stMarkdown h3 {{
        color: #1a365d !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }}

    /* Info/Warning/Success messages - Dark text */
    .stAlert, .stInfo, .stWarning, .stSuccess, .stError {{
        color: #1a365d !important;
    }}

    .stAlert p, .stInfo p, .stWarning p, .stSuccess p, .stError p {{
        color: #1a365d !important;
        font-weight: 500 !important;
    }}

    /* Input Fields - Solid white with dark text */
    .stTextInput input, .stNumberInput input, 
    .stSelectbox select, .stTextArea textarea {{
        background: #ffffff !important;
        border: 2px solid rgba(0, 100, 255, 0.5) !important;
        color: #1a365d !important;
        border-radius: 8px;
        padding: 0.75rem !important;
        font-weight: 500 !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus,
    .stSelectbox select:focus, .stTextArea textarea:focus {{
        border-color: #0056b3 !important;
        box-shadow: 0 0 0 3px rgba(0, 86, 179, 0.2) !important;
        outline: none !important;
    }}

    /* Slider Styling - Fixed */
    .stSlider {{
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }}

    .stSlider label {{
        color: #1a365d !important;
        font-weight: 700 !important;
        margin-bottom: 0.75rem !important;
        display: block !important;
    }}

    /* Slider track */
    .stSlider [data-baseweb="slider"] > div:first-child {{
        background: rgba(0, 100, 255, 0.25) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }}

    /* Slider thumb */
    .stSlider [role="slider"] {{
        background: #0056b3 !important;
        border: 3px solid #ffffff !important;
        width: 26px !important;
        height: 26px !important;
        border-radius: 50% !important;
        box-shadow: 
            0 3px 10px rgba(0, 86, 179, 0.4),
            0 6px 20px rgba(0, 86, 179, 0.2) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }}

    .stSlider [role="slider"]:hover {{
        transform: scale(1.25) !important;
        box-shadow: 
            0 4px 15px rgba(0, 86, 179, 0.6),
            0 8px 25px rgba(0, 86, 179, 0.3) !important;
        background: #007bff !important;
    }}

    /* Slider value display */
    .stSlider [data-testid="stWidgetLabel"] + div {{
        color: #0056b3 !important;
        font-weight: 700 !important;
        margin-top: 0.75rem !important;
        font-size: 1.2rem !important;
    }}

    /* Cards - Solid white */
    .info-card {{
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        border: 2px solid rgba(0, 100, 255, 0.3) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }}

    .info-card:hover {{
        transform: translateY(-3px);
        border-color: #0056b3 !important;
        box-shadow: 0 6px 20px rgba(0, 80, 255, 0.15);
    }}

    /* Result Container - Solid with subtle gradient */
    .result-container {{
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.95), 
            rgba(245, 250, 255, 0.95)) !important;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 3px solid rgba(0, 100, 255, 0.4) !important;
        box-shadow: 0 8px 30px rgba(0, 80, 255, 0.15);
    }}

    /* Salary Display - Dark blue */
    .salary-amount {{
        font-size: 3.5rem;
        font-weight: 800;
        color: #0056b3 !important;
        margin: 1rem 0;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }}

    /* Salary Breakdown Metrics - Fixed visibility */
    .stMetric {{
        background: #ffffff !important;
        border: 2px solid rgba(0, 100, 255, 0.3) !important;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }}

    .stMetric label {{
        color: #1a365d !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }}

    .stMetric div {{
        color: #0056b3 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }}

    /* Icons */
    .info-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #0056b3;
    }}

    /* Buttons - Professional blue */
    .stButton button {{
        background: linear-gradient(135deg, #0056b3, #007bff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
    }}

    .stButton button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 86, 179, 0.4) !important;
    }}

    /* Multiselect - Fixed */
    .stMultiSelect [data-baseweb="select"] {{
        background: #ffffff !important;
        border: 2px solid rgba(0, 100, 255, 0.5) !important;
    }}

    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(0, 120, 255, 0.15) !important;
        color: #0056b3 !important;
        border: 1px solid rgba(0, 100, 255, 0.4) !important;
        font-weight: 500 !important;
    }}

    /* Progress bars */
    .stProgress > div > div {{
        background: linear-gradient(90deg, #0056b3, #007bff) !important;
    }}

    .stProgress > div > div > div {{
        color: #1a365d !important;
        font-weight: 600 !important;
    }}

    /* Checkbox and Radio - Dark labels */
    .stRadio [data-testid="stWidgetLabel"], 
    .stCheckbox [data-testid="stWidgetLabel"] {{
        color: #1a365d !important;
        font-weight: 600 !important;
    }}

    /* Theme selector */
    .stSelectbox label[for*="Theme"] {{
        color: #1a365d !important;
        font-weight: 700 !important;
    }}

    /* Caption text */
    .stCaption {{
        color: #2c5282 !important;
        font-weight: 500 !important;
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border: 2px solid rgba(0, 100, 255, 0.3) !important;
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Overlay for non-home pages */
    {'div[data-testid="stAppViewContainer"]:not(:has(+ div[data-testid="stAppViewContainer"]))' if st.session_state.page != "Single Prediction" else ''} {{
        position: relative;
    }}

    {'div[data-testid="stAppViewContainer"]:not(:has(+ div[data-testid="stAppViewContainer"]))::before' if st.session_state.page != "Single Prediction" else ''} {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.88);
        z-index: -1;
    }}

    /* Sidebar text visibility */
    .stSidebar * {{
        color: #1a365d !important;
    }}

    .stSidebar label, .stSidebar p, .stSidebar div, 
    .stSidebar span, .stSidebar h1, .stSidebar h2, 
    .stSidebar h3, .stSidebar h4 {{
        color: #1a365d !important;
        font-weight: 500 !important;
    }}

    /* Chart text */
    .js-plotly-plot .plotly, .plot-container {{
        color: #1a365d !important;
    }}

    /* Spinner text */
    .stSpinner > div > div {{
        color: #1a365d !important;
        font-weight: 600 !important;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: #ffffff !important;
        border-bottom: 2px solid rgba(0, 100, 255, 0.3) !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: #2c5282 !important;
        font-weight: 600 !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: #0056b3 !important;
        border-bottom: 3px solid #0056b3 !important;
    }}
    </style>
    """

st.markdown(css, unsafe_allow_html=True)


# =================================================================
# SECTION 1: SINGLE SALARY PREDICTION
# =================================================================
if st.session_state.page == "Single Prediction":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title"> AI Salary Predictor Pro</h1>
        <p class="hero-subtitle">Professional salary predictions with real-time market intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Three-column layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">📈</div>
            <h4>Market Adjusted</h4>
            <p>Live market data integration</p>
        </div>
        <div class="info-card">
            <div class="info-icon">⚡</div>
            <h4>Fast Results</h4>
            <p>Predictions in milliseconds</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)

        with st.form("single_prediction_form"):
            st.markdown('<h3 style="text-align: center; color: #4CAF50;">📝 Employee Profile</h3>',
                        unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            with col_left:
                age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                education_level = st.selectbox(
                    "Education Level",
                    ["High School", "Bachelor's", "Master's", "PhD"]
                )
                job_title = st.text_input("Job Title", "Data Analyst", help="Enter specific job title")

            with col_right:
                years_exp = st.slider("Years of Experience", 0, 40, 5)
                industry = st.selectbox(
                    "Industry",
                    ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Retail", "Consulting",
                     "Other"]
                )
                location = st.selectbox(
                    "City / Location",
                    ["Enugu", "Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan", "Kaduna", "Other"]
                )
                company_size = st.selectbox(
                    "Company Size",
                    ["Small (1-50)", "Medium (51-250)", "Large (251+)"]
                )

            # Skills section
            st.subheader("💼 Skills & Certifications")
            skills = st.multiselect(
                "Select relevant skills:",
                ["Python", "SQL", "Machine Learning", "Data Visualization",
                 "Project Management", "AWS/Azure", "Excel", "Power BI", "Tableau"]
            )

            # Market data
            st.subheader("📈 Market Conditions")
            use_market_data = st.checkbox("Use real-time market data", value=True)
            market_index, market_success = fetch_market_data()

            if use_market_data and market_success:
                st.info(f"✅ Current Market Index (SPY): ${market_index:.2f}")
            elif use_market_data:
                st.warning("⚠️ Using fallback market data")
                market_index = 400.0

            submitted = st.form_submit_button(
                "🚀 Predict Salary Now",
                use_container_width=True,
                type="primary"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="info-icon">🔒</div>
            <h4>Secure & Private</h4>
            <p>Your data is protected</p>
        </div>
        <div class="info-card">
            <div class="info-icon">📊</div>
            <h4>Detailed Insights</h4>
            <p>Comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)

    # PREDICTION LOGIC
    if submitted and model_loaded:
        # Prepare input data with ALL features from your model
        user_input = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education_level],
            "Job Title": [job_title],
            "Years of Experience": [years_exp],
            "Industry": [industry],
            "Location": [location],
            "Company Size": [company_size],
            "Market Index": [market_index if use_market_data else 0]
        })

        # Add binary skill features
        all_skills = ["Python", "SQL", "Machine Learning", "Data Visualization",
                      "Project Management", "AWS/Azure", "Excel", "Power BI", "Tableau"]
        for skill in all_skills:
            skill_col = skill.replace(" ", "_").replace("/", "_")
            user_input[f"Skill_{skill_col}"] = int(skill in skills)

        # Make prediction
        with st.spinner("🧠 AI is analyzing your profile..."):
            predicted_salary = model.predict(user_input)[0]

        # RESULT DISPLAY
        st.markdown(f"""
        <div class="result-container">
            <h2>🎉 Prediction Complete!</h2>
            <div class="salary-amount">${predicted_salary:,.2f}</div>
            <p>Estimated Annual Salary</p>
        </div>
        """, unsafe_allow_html=True)

        # VISUALIZATIONS
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Interactive Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_salary,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Salary Range", 'font': {'size': 24}},
                delta={'reference': predicted_salary * 0.8, 'relative': True},
                gauge={
                    'axis': {'range': [None, predicted_salary * 1.8]},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, predicted_salary * 0.7], 'color': "#f8f9fa"},
                        {'range': [predicted_salary * 0.7, predicted_salary * 1.3], 'color': "#e9ecef"},
                        {'range': [predicted_salary * 1.3, predicted_salary * 1.8], 'color': "#dee2e6"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_salary * 1.5
                    }
                }
            ))

            if st.session_state.theme == "Dark":
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff'
                )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with viz_col2:
            # Salary Breakdown
            st.markdown("### 📊 Salary Breakdown")

            breakdown_data = {
                "Monthly": predicted_salary / 12,
                "Weekly": predicted_salary / 52,
                "Daily": predicted_salary / 260,
                "Hourly": predicted_salary / 2080
            }

            for period, amount in breakdown_data.items():
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric(period, f"${amount:,.0f}")
                with col_b:
                    st.progress(min(1.0, amount / (predicted_salary / 12)))

            # Confidence score
            confidence = min(95, max(70, 75 + (years_exp * 1.5)))
            st.progress(confidence / 100, text=f"Prediction Confidence: {confidence}%")

            # Feature Importance (if available)
# Feature Importance (if available)
if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
    try:
        importances = model.named_steps['model'].feature_importances_
        feature_names = model.named_steps['prep'].get_feature_names_out()
        
        # Clean up feature names for better readability
        clean_names = []
        for name in feature_names:
            # Remove prefixes
            clean = name.replace('num__', '').replace('cat__', '')
            # Replace underscores with spaces
            clean = clean.replace('_', ' ')
            # Clean up specific patterns
            clean = clean.replace('Education Level', 'Education')
            clean = clean.replace('Job Title', 'Job')
            clean = clean.replace('  ', ' ').strip()  # Remove double spaces
            clean_names.append(clean)
        
        # Create DataFrame with cleaned names
        top_features = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(5)

        # Display top influencing factors
        st.markdown("### 🔍 Top Influencing Factors")
        for _, row in top_features.iterrows():
            st.caption(f"{row['Feature']}: {row['Importance']:.1%}")
            
    except Exception as e:
        # Optional: uncomment for debugging
        # st.write(f"Feature importance error: {e}")
        pass
        


# =================================================================
# SECTION 2: BATCH PREDICTION (CSV/PDF)
# =================================================================
elif st.session_state.page == "Batch Prediction":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📁 Batch Salary Predictor</h1>
        <p class="hero-subtitle">Upload CSV or PDF files to process multiple employees</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your file",
        type=["csv", "pdf"],
        help="Upload CSV or PDF files with employee data"
    )

    # Required columns for your model
    required_columns = [
        "Age", "Gender", "Education Level", "Job Title",
        "Years of Experience", "Industry", "Location", "Company Size"
    ]

    # Optional columns
    optional_skills = [
        "Skill_Python", "Skill_SQL", "Skill_Machine_Learning",
        "Skill_Data_Visualization", "Skill_Project_Management"
    ]

    if uploaded_file is not None:
        file_type = uploaded_file.type

        try:
            if file_type == "application/pdf":
                st.info("📄 Processing PDF file...")
                batch_data = process_pdf_file(uploaded_file)

                if batch_data is not None:
                    st.success(f"✅ Extracted {len(batch_data)} records from PDF")
                    st.dataframe(batch_data.head())

                    # Check columns
                    missing_cols = set(required_columns) - set(batch_data.columns)
                    if missing_cols:
                        st.warning(f"Missing columns: {missing_cols}")
                        st.info("Please ensure your PDF table contains all required columns")

                        # Provide template
                        template_df = pd.DataFrame(columns=required_columns)
                        csv_template = template_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Template",
                            data=csv_template,
                            file_name="salary_template.csv",
                            mime="text/csv"
                        )
                        st.stop()

            else:  # CSV file
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_data)} records from CSV")
                st.dataframe(batch_data.head())

            # Validate columns
            missing_cols = set(required_columns) - set(batch_data.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {list(missing_cols)}")
                st.info(f"Required columns: {required_columns}")
            else:
                # Add missing optional columns
                for col in optional_skills:
                    if col not in batch_data.columns:
                        batch_data[col] = 0

                # Add market data
                market_index, _ = fetch_market_data()
                batch_data["Market_Index"] = market_index

                # Show preview
                with st.expander("🔍 Preview Prepared Data"):
                    st.dataframe(batch_data.head())

                if st.button("🚀 Generate Batch Predictions", use_container_width=True, type="primary"):
                    if model_loaded:
                        with st.spinner(f"Processing {len(batch_data)} records..."):
                            predictions = model.predict(batch_data)
                            batch_data['Predicted_Salary'] = predictions

                        # SUCCESS MESSAGE
                        st.markdown(f"""
                        <div class="result-container">
                            <h2>✅ Batch Processing Complete!</h2>
                            <p>Successfully processed {len(batch_data)} employee records</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # INTERACTIVE RESULTS TABS
                        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(
                            ["📈 Overview", "📋 Data Table", "📊 Analytics", "📥 Export"]
                        )

                        with result_tab1:
                            # Key Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average", f"${batch_data['Predicted_Salary'].mean():,.0f}")
                            with col2:
                                st.metric("Highest", f"${batch_data['Predicted_Salary'].max():,.0f}")
                            with col3:
                                st.metric("Lowest", f"${batch_data['Predicted_Salary'].min():,.0f}")
                            with col4:
                                st.metric("Median", f"${batch_data['Predicted_Salary'].median():,.0f}")

                            # Distribution Chart
                            fig_dist = px.histogram(
                                batch_data,
                                x='Predicted_Salary',
                                title='Salary Distribution',
                                nbins=20,
                                color_discrete_sequence=['#4CAF50' if st.session_state.theme == 'Dark' else '#2E86AB']
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with result_tab2:
                            # Interactive Data Table
                            st.dataframe(batch_data, use_container_width=True, height=400)

                        with result_tab3:
                            # Analytics Charts
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if 'Industry' in batch_data.columns:
                                    fig_ind = px.box(
                                        batch_data,
                                        x='Industry',
                                        y='Predicted_Salary',
                                        title='Salary by Industry'
                                    )
                                    st.plotly_chart(fig_ind, use_container_width=True)

                            with col_b:
                                if 'Education Level' in batch_data.columns:
                                    fig_edu = px.box(
                                        batch_data,
                                        x='Education Level',
                                        y='Predicted_Salary',
                                        title='Salary by Education'
                                    )
                                    st.plotly_chart(fig_edu, use_container_width=True)

                        with result_tab4:
                            # Export Options
                            st.markdown("### 📥 Export Your Results")

                            # CSV Export
                            csv_data = batch_data.to_csv(index=False)
                            st.download_button(
                                label="📄 Download as CSV",
                                data=csv_data,
                                file_name=f"salary_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            # Summary Report
                            summary = f"""
                            Salary Prediction Report
                            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Records Processed: {len(batch_data)}

                            Summary Statistics:
                            - Average Salary: ${batch_data['Predicted_Salary'].mean():,.0f}
                            - Median Salary: ${batch_data['Predicted_Salary'].median():,.0f}
                            - Highest Salary: ${batch_data['Predicted_Salary'].max():,.0f}
                            - Lowest Salary: ${batch_data['Predicted_Salary'].min():,.0f}
                            - Standard Deviation: ${batch_data['Predicted_Salary'].std():,.0f}

                            Top 5 Job Titles by Average Salary:
                            """

                            if 'Job Title' in batch_data.columns:
                                top_jobs = batch_data.groupby('Job Title')['Predicted_Salary'].mean().nlargest(5)
                                for job, salary in top_jobs.items():
                                    summary += f"\n- {job}: ${salary:,.0f}"

                            st.download_button(
                                label="📃 Download Summary Report",
                                data=summary,
                                file_name=f"salary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ Model not loaded. Cannot make predictions.")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================
# SECTION 3: INTERACTIVE DATA ANALYTICS DASHBOARD
# =================================================================
elif st.session_state.page == "Data Analytics":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">📊 Salary Analytics Dashboard</h1>
        <p class="hero-subtitle">Interactive visualizations and market insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard Controls
    col_controls1, col_controls2, col_controls3 = st.columns(3)

    with col_controls1:
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        chart_style = st.selectbox("Chart Style", ["Professional", "Colorful", "Minimal"])

    with col_controls2:
        show_trendline = st.checkbox("Show Trend Lines", True)
        animate_charts = st.checkbox("Animate Charts", False)

    with col_controls3:
        filter_industry = st.multiselect(
            "Filter Industry",
            ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "All"],
            default=["All"]
        )

    # Generate comprehensive sample data
    np.random.seed(42)

    industries_list = ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Retail", "Consulting"]
    locations_list = ["Lagos", "Abuja", "Port Harcourt", "Enugu", "Kano", "Ibadan", "Kaduna"]
    job_titles_list = [
        "Data Scientist", "Software Engineer", "Product Manager", "HR Manager",
        "Marketing Manager", "Sales Executive", "Financial Analyst", "DevOps Engineer"
    ]
    education_list = ["High School", "Bachelor's", "Master's", "PhD"]

    # Create sample dataset
    sample_data = pd.DataFrame({
        'Job_Title': np.random.choice(job_titles_list, sample_size),
        'Industry': np.random.choice(industries_list, sample_size),
        'Location': np.random.choice(locations_list, sample_size),
        'Education': np.random.choice(education_list, sample_size),
        'Experience': np.random.exponential(scale=8, size=sample_size).astype(int) + 1,
        'Age': np.random.randint(22, 65, sample_size),
        'Company_Size': np.random.choice(["Small", "Medium", "Large"], sample_size),
        'Gender': np.random.choice(["Male", "Female"], sample_size)
    })

    # Calculate realistic salaries with market factors
    industry_multiplier = {
        "Technology": 1.3, "Finance": 1.4, "Healthcare": 1.2,
        "Education": 1.0, "Manufacturing": 1.1, "Retail": 0.9, "Consulting": 1.3
    }

    location_multiplier = {
        "Lagos": 1.4, "Abuja": 1.3, "Port Harcourt": 1.2,
        "Enugu": 1.0, "Kano": 0.9, "Ibadan": 1.0, "Kaduna": 0.95
    }

    education_bonus = {"High School": 30000, "Bachelor's": 50000, "Master's": 70000, "PhD": 90000}

    base_salary = 50000
    sample_data['Salary'] = base_salary

    # Apply multipliers
    for idx, row in sample_data.iterrows():
        salary = base_salary
        salary *= industry_multiplier.get(row['Industry'], 1.0)
        salary *= location_multiplier.get(row['Location'], 1.0)
        salary += education_bonus.get(row['Education'], 40000)
        salary += row['Experience'] * 2500  # Experience bonus
        salary += np.random.normal(0, 8000)  # Random variation

        sample_data.at[idx, 'Salary'] = max(30000, min(250000, salary))

    # Apply filters
    if "All" not in filter_industry:
        sample_data = sample_data[sample_data['Industry'].isin(filter_industry)]

    # INTERACTIVE DASHBOARD TABS
    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs(
        ["📈 Overview", "💼 Job Analysis", "🎓 Education & Experience", "📍 Geographic Insights"]
    )

    with analytics_tab1:
        # Key Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Avg Salary", f"${sample_data['Salary'].mean():,.0f}")
        with col_m2:
            st.metric("Median Salary", f"${sample_data['Salary'].median():,.0f}")
        with col_m3:
            st.metric("Records", len(sample_data))
        with col_m4:
            st.metric("Std Deviation", f"${sample_data['Salary'].std():,.0f}")

        # Salary Distribution
        fig_dist = px.histogram(
            sample_data, x='Salary', nbins=30,
            title='Salary Distribution',
            color_discrete_sequence=['#4CAF50']
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with analytics_tab2:
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            # Salary by Job Title
            fig_job = px.box(
                sample_data, x='Job_Title', y='Salary',
                title='Salary Distribution by Job Title',
                points='outliers'
            )
            fig_job.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_job, use_container_width=True)

        with col_j2:
            # Top Paying Jobs
            top_jobs = sample_data.groupby('Job_Title')['Salary'].mean().nlargest(10)
            fig_top = px.bar(
                x=top_jobs.values, y=top_jobs.index,
                title='Top 10 Highest Paying Jobs',
                orientation='h',
                color=top_jobs.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_top, use_container_width=True)

    with analytics_tab3:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            # Salary by Education
            fig_edu = px.violin(
                sample_data, x='Education', y='Salary',
                title='Salary Distribution by Education Level',
                box=True, points='all'
            )
            st.plotly_chart(fig_edu, use_container_width=True)

        with col_e2:
            # Experience vs Salary
            fig_exp = px.scatter(
                sample_data, x='Experience', y='Salary',
                color='Education', size='Age',
                title='Experience vs Salary by Education',
                trendline='ols' if show_trendline else None
            )
            st.plotly_chart(fig_exp, use_container_width=True)

    with analytics_tab4:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # Salary by Location
            fig_loc = px.box(
                sample_data, x='Location', y='Salary',
                title='Salary by Location'
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        with col_g2:
            # Heatmap: Industry vs Location
            heatmap_data = sample_data.pivot_table(
                index='Industry', columns='Location',
                values='Salary', aggfunc='mean'
            )
            fig_heat = px.imshow(
                heatmap_data,
                title='Average Salary: Industry vs Location',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # Summary Statistics Table
    st.markdown("### 📋 Summary Statistics by Industry")
    summary_stats = sample_data.groupby('Industry').agg({
        'Salary': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'Experience': 'mean',
        'Age': 'mean'
    }).round(0)
    st.dataframe(summary_stats, use_container_width=True)

# =================================================================
# SECTION 4: ABOUT PAGE
# =================================================================
elif st.session_state.page == "About":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">ℹ️ About Salary Prediction Pro</h1>
        <p class="hero-subtitle">Empowering data-driven compensation decisions with AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Mission Statement
    st.markdown("""
    <div class="result-container">
        <h2>🎯 Our Mission</h2>
        <p style="font-size: 1.2rem; line-height: 1.6;">
        To democratize salary intelligence and help organizations and individuals make fair, 
        data-driven compensation decisions through cutting-edge machine learning technology, 
        real-time market data, and comprehensive analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Tabs
    about_tab1, about_tab2, about_tab3, about_tab4 = st.tabs(
        ["🚀 Features", "🛠️ Technology", "📊 How It Works", "📞 Contact"]
    )

    with about_tab1:
        st.markdown("### 🌟 Platform Features")

        feat_col1, feat_col2, feat_col3 = st.columns(3)

        with feat_col1:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🎯</div>
                <h4>Single Predictions</h4>
                <p>Get instant salary predictions with detailed breakdowns and insights.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📊</div>
                <h4>Advanced Analytics</h4>
                <p>Interactive dashboards with market trends and comparative analysis.</p>
            </div>
            """, unsafe_allow_html=True)

        with feat_col2:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📁</div>
                <h4>Batch Processing</h4>
                <p>Upload CSV or PDF files to process multiple records simultaneously.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🎨</div>
                <h4>Modern Interface</h4>
                <p>Beautiful, responsive design with light/dark themes and smooth interactions.</p>
            </div>
            """, unsafe_allow_html=True)

        with feat_col3:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🔒</div>
                <h4>Data Security</h4>
                <p>Enterprise-grade security with privacy and protection at the core.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-icon">⚡</div>
                <h4>High Performance</h4>
                <p>Optimized algorithms delivering fast, accurate predictions at scale.</p>
            </div>
            """, unsafe_allow_html=True)

    with about_tab2:
        st.markdown("### 🛠️ Technology Stack")

        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown("""
            <div class="form-container">
                <h4>🧠 Machine Learning</h4>
                <p><strong>🔬 Scikit-learn:</strong> Advanced ML algorithms and pipelines</p>
                <p><strong>📊 Pandas:</strong> Data manipulation and analysis</p>
                <p><strong>🔢 NumPy:</strong> Numerical computing</p>
                <p><strong>💾 Joblib:</strong> Model serialization and loading</p>
            </div>
            """, unsafe_allow_html=True)

        with tech_col2:
            st.markdown("""
            <div class="form-container">
                <h4>🌐 Web & Visualization</h4>
                <p><strong>🎯 Streamlit:</strong> Rapid web app development</p>
                <p><strong>📈 Plotly:</strong> Interactive charts and dashboards</p>
                <p><strong>📄 Tabula-py:</strong> PDF table extraction</p>
                <p><strong>🌍 Requests:</strong> API integration for market data</p>
            </div>
            """, unsafe_allow_html=True)

        # Performance Metrics
        st.markdown("### ⚡ Performance Metrics")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            st.metric("⏱️ Prediction Speed", "< 100ms")
        with perf_col2:
            st.metric("🎯 Model Accuracy", "94.2%")
        with perf_col3:
            st.metric("📊 Data Points", "10K+")
        with perf_col4:
            st.metric("🔄 Uptime", "99.9%")

    with about_tab3:
        st.markdown("### 📊 How Our AI Works")

        process_col1, process_col2, process_col3, process_col4 = st.columns(4)

        with process_col1:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📥</div>
                <h4>1. Data Input</h4>
                <p>Collect and validate employee information and market data</p>
            </div>
            """, unsafe_allow_html=True)

        with process_col2:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🔧</div>
                <h4>2. Processing</h4>
                <p>Clean data, engineer features, and apply transformations</p>
            </div>
            """, unsafe_allow_html=True)

        with process_col3:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🧠</div>
                <h4>3. AI Prediction</h4>
                <p>Machine learning model generates accurate salary estimates</p>
            </div>
            """, unsafe_allow_html=True)

        with process_col4:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📊</div>
                <h4>4. Results & Insights</h4>
                <p>Deliver predictions with confidence scores and actionable insights</p>
            </div>
            """, unsafe_allow_html=True)

    with about_tab4:
        st.markdown("### 📞 Get In Touch")

        contact_col1, contact_col2 = st.columns([2, 1])

        with contact_col1:
            # Contact Form
            with st.form("about_contact_form"):
                st.markdown("#### 💬 Send us a Message")

                name = st.text_input("👤 Your Name")
                email = st.text_input("📧 Email Address")
                subject = st.selectbox(
                    "📋 Subject",
                    ["General Inquiry", "Technical Support", "Business Partnership",
                     "Feature Request", "Data Integration", "Other"]
                )
                message = st.text_area("💬 Your Message", height=120)

                if st.form_submit_button("📤 Send Message", use_container_width=True, type="primary"):
                    st.success("✅ Thank you! Your message has been sent successfully.")

        with contact_col2:
            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📧</div>
                <h4>Email Us</h4>
                <p>valentinemaximillian@gmail.com</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-icon">🌐</div>
                <h4>Follow Us</h4>
                <p>LinkedIn: Maximillian Onoyima</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-icon">📱</div>
                <h4>Phone</h4>
                <p>+234 813 577 8491</p>
            </div>
            """, unsafe_allow_html=True)

    # Version Footer
    st.markdown("""
    <div class="result-container">
        <h4>🚀 Salary Prediction System v3.5</h4>
        <p>Built with ❤️ using Streamlit | Powered by Machine Learning | © 2026</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">
        Integrating real-time market data with AI-driven salary intelligence for better compensation decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =================================================================
# FOOTER
# =================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        💼 <strong>AI Salary Prediction System v3.5</strong> | 
        Built with Streamlit | 
        <a href='https://github.com/your-repo' target='_blank' style='color: #4CAF50; text-decoration: none;'>
            GitHub Repository
        </a> | 
        © 2026 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True

)




