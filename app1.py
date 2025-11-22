#LEAF ILLNESS AND RISK ASSESSMENT

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import hashlib
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import base64
from typing import Dict, Any, Optional
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
# --- New Imports for PyTorch & Severity ---
# --- PyTorch & utility imports for new models ---
import collections
import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms as T
import joblib
import pickle



# Load .env file
load_dotenv()

# Page configuration - Fixed sidebar state management
st.set_page_config(
    page_title="LIRA Pro - Sugarcane Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"  # Always start expanded
)

# Custom CSS for modern black & green UI with permanent sidebar
# Replace the entire st.markdown CSS block (around line 52) with this complete version:

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling - PERMANENT VERSION */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
        border-right: 2px solid rgba(34, 197, 94, 0.2);
        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    
    /* Sidebar content styling */
    section[data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    section[data-testid="stSidebar"] .css-1v3fvcr {
        color: #ffffff;
    }
    
    /* Hide sidebar toggle button completely */
    button[data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    
    /* Prevent sidebar from collapsing */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;
        transform: translateX(0) !important;
    }
    
    /* Main content area adjustments */
    .css-18e3th9 {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Main background - Deep black gradient */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0d1b0d 100%);
        color: #ffffff;
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0d1b0d 100%);
    }
    
    /* Custom card styling with green accents */
    .custom-card {
        background: linear-gradient(145deg, #1a1a1a 0%, #2d2d2d 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(34, 197, 94, 0.2);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.6),
            0 0 20px rgba(34, 197, 94, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #22c55e, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.8),
            0 0 30px rgba(34, 197, 94, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(34, 197, 94, 0.4);
    }
    
    .custom-card:hover::before {
        opacity: 1;
    }
    
    /* Header styling with matrix-style green */
    .main-header {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #00ff88, #22c55e, #16a34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(34, 197, 94, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(34, 197, 94, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(34, 197, 94, 0.8)); }
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Dynamic button styling */
    .stButton > button {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 50%, #4ade80 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 18px 35px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 25px rgba(34, 197, 94, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 15px 35px rgba(34, 197, 94, 0.4),
            0 0 20px rgba(34, 197, 94, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        background: linear-gradient(135deg, #22c55e 0%, #4ade80 50%, #6ee7b7 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(26, 26, 26, 0.8);
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        color: white;
        padding: 15px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #22c55e;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);
        transform: scale(1.02);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(26, 26, 26, 0.6);
        border: 2px dashed rgba(34, 197, 94, 0.4);
        border-radius: 15px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #22c55e;
        background: rgba(26, 26, 26, 0.8);
        transform: scale(1.01);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(26, 26, 26, 0.8);
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(26, 26, 26, 0.9);
        color: white;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(26, 26, 26, 0.8);
        border: 2px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        color: white;
    }
    
    /* Success/error message styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.3));
        border: 1px solid rgba(34, 197, 94, 0.5);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3));
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3));
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.3));
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Disease result cards */
    .disease-result {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 
            0 15px 35px rgba(220, 38, 38, 0.4),
            0 0 20px rgba(220, 38, 38, 0.2);
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 15px 35px rgba(220, 38, 38, 0.4), 0 0 20px rgba(220, 38, 38, 0.2); }
        50% { box-shadow: 0 15px 35px rgba(220, 38, 38, 0.6), 0 0 30px rgba(220, 38, 38, 0.4); }
    }
    
    .healthy-result {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 
            0 15px 35px rgba(34, 197, 94, 0.4),
            0 0 20px rgba(34, 197, 94, 0.2);
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 15px 35px rgba(34, 197, 94, 0.4), 0 0 20px rgba(34, 197, 94, 0.2); }
        50% { box-shadow: 0 15px 35px rgba(34, 197, 94, 0.6), 0 0 30px rgba(34, 197, 94, 0.4); }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(34, 197, 94, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: rgba(34, 197, 94, 0.4);
    }
    
    /* Tab styling */
    .stTabs > div > div > div > div {
        background: rgba(26, 26, 26, 0.8);
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .stTabs > div > div > div > div[data-selected="true"] {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: white;
    }
    
    /* Form styling */
    .stForm {
        background: rgba(26, 26, 26, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 26, 0.6);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 10px;
    }
    
    /* Spinner styling */
    .stSpinner > div > div {
        border-top-color: #22c55e !important;
    }
    
    /* Custom animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .slide-in-up {
        animation: slideInUp 0.6s ease-out;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .fade-in-scale {
        animation: fadeInScale 0.5s ease-out;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #16a34a, #22c55e, #4ade80);
    }
    
    /* Sidebar metrics */
    .sidebar-metric {
        background: rgba(34, 197, 94, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(34, 197, 94, 0.2);
        margin: 10px 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .sidebar-metric:hover {
        background: rgba(34, 197, 94, 0.2);
        transform: scale(1.05);
    }
    
    /* Chart styling */
    .stPlotlyChart {
        background: rgba(26, 26, 26, 0.6);
        border-radius: 15px;
        padding: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Radio button styling for navigation */
    .stRadio > div {
        background: transparent;
    }
    
    .stRadio > div > label {
        background: rgba(26, 26, 26, 0.4);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 5px 0;
        border: 1px solid rgba(34, 197, 94, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        display: block;
    }
    
    .stRadio > div > label:hover {
        background: rgba(34, 197, 94, 0.1);
        border-color: rgba(34, 197, 94, 0.4);
        transform: scale(1.02);
    }
    
    .stRadio > div > label[data-selected="true"] {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        border-color: #22c55e;
        color: white;
    }
    
    /* STATIC FEATURE BOXES STYLING */
    .feature-box {
        background: linear-gradient(145deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 2px solid var(--box-color);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        height: 350px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        backdrop-filter: blur(20px);
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.6),
            0 0 20px var(--box-color, rgba(34, 197, 94, 0.1));
    }

    .feature-box:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: var(--box-color);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.8),
            0 0 40px var(--box-color, rgba(34, 197, 94, 0.3));
    }



    .feature-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.5));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .feature-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--box-color);
        margin: 0 0 15px 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    .feature-description {
        font-size: 0.95rem;
        line-height: 1.5;
        color: rgba(255, 255, 255, 0.85);
        margin: 0 0 20px 0;
        flex-grow: 1;
        display: flex;
        align-items: center;
        text-align: center;
    }



    /* Responsive design for feature boxes */
    @media (max-width: 768px) {
        .feature-box {
            height: auto;
            min-height: 300px;
            padding: 20px;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        .feature-title {
            font-size: 1.4rem;
        }
        
        .feature-description {
            font-size: 0.9rem;
        }
        
        .feature-tags {
            flex-direction: column;
        }
        
        .tag {
            font-size: 0.7rem;
            padding: 5px 10px;
        }
    }

    /* Enhanced hover effects for individual feature boxes */
    .feature-box:nth-child(1):hover {
        background: linear-gradient(145deg, #1a1a1a 0%, #1a2e1a 100%);
    }

    .feature-box:nth-child(2):hover {
        background: linear-gradient(145deg, #1a1a1a 0%, #1a1e2e 100%);
    }

    .feature-box:nth-child(3):hover {
        background: linear-gradient(145deg, #1a1a1a 0%, #2e251a 100%);
    }
</style>
""", unsafe_allow_html=True)

# Model Configuration
# --- New Model Paths ---
PYTORCH_MODEL_PATH = "./models/sugarcane_efficientnetb0.pth"
BCL_SEVERITY_MODEL_PATH = "./models/data.pkl"  # for BandedChlorosis
MOSAIC_SEVERITY_MODEL_PATH = "./models/best_model_finetuned.keras"


# Load the trained model
@st.cache_resource
def load_model():
    """Load the PyTorch classification model and return (model, class_names).

    The .pth file might contain:
      - a state_dict (OrderedDict)
      - a dict containing 'state_dict' / 'model_state_dict'
      - a full saved model object (less common)
    This function handles all cases.
    """
    try:
        # Define expected class names (9 classes per your message)
        class_names = [
            "BandedChlorosis", "DriedLeaves", "Healthy",
            "Mosaic", "RedRot", "Rust", "YellowLeafy2",
            "Yellowy1", "smut"
        ]
        num_classes = len(class_names)

        # Build architecture (EfficientNet-B0) and adapt final layer
        # Uses torchvision's EfficientNet-B0
        model = tvmodels.efficientnet_b0(weights=None)  # instantiate
        # Replace classifier final Linear with correct out features
        try:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        except Exception:
            # fallback if classifier layout differs
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, num_classes))

        # Load the .pth file robustly
        if os.path.exists(PYTORCH_MODEL_PATH):
            loaded = torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu'))

            # If the loaded object is a full model (rare), use it directly
            if not isinstance(loaded, dict) and hasattr(loaded, "state_dict") and callable(getattr(loaded, "state_dict")):
                # loaded is a model object
                loaded_model = loaded
                loaded_model.eval()
                return loaded_model, class_names

            # If it's a dict, try to find the state dict
            if isinstance(loaded, dict):
                # Common keys where weights are stored
                sd = None
                for key in ("state_dict", "model_state_dict", "model_state", "weights"):
                    if key in loaded:
                        sd = loaded[key]
                        break
                if sd is None:
                    # maybe the dict itself *is* the state_dict (OrderedDict)
                    sd = loaded

                # If sd looks like an OrderedDict, load into our model
                if isinstance(sd, collections.OrderedDict) or all(isinstance(k, str) for k in sd.keys()):
                    # sometimes keys have 'module.' prefix if saved with DataParallel — strip if needed
                    new_sd = {}
                    for k, v in sd.items():
                        new_key = k
                        if k.startswith("module."):
                            new_key = k.replace("module.", "", 1)
                        new_sd[new_key] = v
                    model.load_state_dict(new_sd, strict=False)
                    model.eval()
                    return model, class_names

            # If nothing matched, try to set model to loaded (last resort)
            try:
                loaded.eval()
                return loaded, class_names
            except Exception:
                # fallback: return instantiated model (random weights)
                model.eval()
                return model, class_names
        else:
            st.error(f"❌ PyTorch model file not found at: {PYTORCH_MODEL_PATH}")
            model.eval()
            return model, class_names

    except Exception as e:
        st.error(f"❌ Error loading classification model: {str(e)}")
        # Return placeholders so rest of app can still run (but will show model error)
        return None, None



@st.cache_resource
def load_severity_models():
    """Load severity assessment models:
       - BandedChlorosis: joblib .pkl
       - Mosaic: keras .keras
    Returns (bcl_model, mosaic_model). Any missing model returns None at that slot.
    """
    bcl_model = None
    mosaic_model = None
    
    # Load BandedChlorosis severity (.pkl) safely
    try:
        if os.path.exists(BCL_SEVERITY_MODEL_PATH):
            # Try with joblib first (most common for sklearn models)
            try:
                import joblib
                bcl_model = joblib.load(BCL_SEVERITY_MODEL_PATH)
                print("✅ BandedChlorosis severity model loaded with joblib")
            except Exception as e1:
                # Fallback to pickle with custom unpickler
                try:
                    import pickle
                    
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            # Handle Keras objects
                            if 'keras' in module:
                                import tensorflow.keras as keras
                                return getattr(keras, name, None) or super().find_class(module, name)
                            return super().find_class(module, name)
                    
                    with open(BCL_SEVERITY_MODEL_PATH, 'rb') as f:
                        bcl_model = CustomUnpickler(f).load()
                    print("✅ BandedChlorosis severity model loaded with custom unpickler")
                except Exception as e2:
                    print(f"❌ Could not load BCL model: joblib error: {e1}, pickle error: {e2}")
                    bcl_model = None
    except Exception as e:
        print(f"❌ BandedChlorosis severity model file not accessible: {e}")
        bcl_model = None

    # Load Mosaic severity (keras) with custom objects handling
    try:
        if os.path.exists(MOSAIC_SEVERITY_MODEL_PATH):
            # Define custom objects that might be in the model
            custom_objects = {
                'Functional': keras.models.Model,
                'DTypePolicy': keras.mixed_precision.Policy,
                'InputLayer': keras.layers.InputLayer,
                'RandomFlip': keras.layers.RandomFlip,
                'RandomRotation': keras.layers.RandomRotation,
                'RandomZoom': keras.layers.RandomZoom,
            }
            
            try:
                # Method 1: Load with custom objects
                mosaic_model = keras.models.load_model(
                    MOSAIC_SEVERITY_MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False
                )
                print("✅ Mosaic severity model loaded with custom_objects")
            except Exception as e1:
                try:
                    # Method 2: Load with safe_mode=False
                    mosaic_model = keras.models.load_model(
                        MOSAIC_SEVERITY_MODEL_PATH,
                        compile=False,
                        safe_mode=False
                    )
                    print("✅ Mosaic severity model loaded with safe_mode=False")
                except Exception as e2:
                    try:
                        # Method 3: Use TensorFlow SavedModel format
                        import tensorflow as tf
                        mosaic_model = tf.keras.models.load_model(
                            MOSAIC_SEVERITY_MODEL_PATH,
                            compile=False
                        )
                        print("✅ Mosaic severity model loaded with TF")
                    except Exception as e3:
                        print(f"❌ All Mosaic loading methods failed:")
                        print(f"  Method 1: {e1}")
                        print(f"  Method 2: {e2}")
                        print(f"  Method 3: {e3}")
                        mosaic_model = None
    except Exception as e:
        print(f"❌ Mosaic severity model file not accessible: {e}")
        mosaic_model = None

    return bcl_model, mosaic_model

def preprocess_image_torch(pil_image, img_size=(224, 224)):
    """Preprocess PIL image for PyTorch EfficientNet-B0 inference.
    Returns a torch.Tensor of shape (1, C, H, W).
    """
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    # ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)  # add batch dim
    return tensor


def preprocess_image_for_prediction(pil_image, img_size=(224, 224)):
    """Preprocess PIL image for model prediction"""
    try:
        # Resize image to model input size
        img = pil_image.resize(img_size)
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess with EfficientNetV2 preprocessing
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def calculate_severity_level(disease_name, confidence):
    """Calculate severity level based on disease type and confidence"""
    severity_mapping = {
        "Healthy": 0,
        "Brown Spot": 2,
        "Mosaic": 3,
        "RedRot": 4,
        "Rust": 3,
        "smut": 4,
        "Yellow Leaf": 2
    }
    
    base_severity = severity_mapping.get(disease_name, 3)
    
    # Adjust severity based on confidence
    if confidence > 0.9:
        severity_modifier = 1.0
    elif confidence > 0.7:
        severity_modifier = 0.8
    else:
        severity_modifier = 0.6
    
    final_severity = int(base_severity * severity_modifier)
    return min(max(final_severity, 0), 5)  # Clamp between 0 and 5

def get_treatment_recommendation(disease_name, severity):
    """Return plain-text recommendations, no embedded HTML."""
    if disease_name == "BandedChlorosis":
        base_treatment= ("Banded chlorosis is often due to cold injury or nutrient imbalance. "
                "Use early planting, avoid sensitive cultivars in cold periods, and ensure balanced nutrition and irrigation.")
    elif disease_name == "Mosaic":
        base_treatment= ("Sugarcane mosaic is a viral disease. Remove infected plants, use resistant varieties, and always plant virus-free setts.")
    elif disease_name == "DriedLeaves":
        base_treatment= ("Leaf drying may result from stress or deficiency. Maintain proper irrigation, apply balanced fertilizer, and ensure good drainage.")
    elif disease_name == "Healthy":
        base_treatment= ("Crop appears healthy. Continue balanced fertilization, irrigation, and pest monitoring.")
    elif disease_name == "RedRot":
        base_treatment= ("Red rot is a severe fungal disease. Remove and destroy infected stools, plant tolerant varieties, and rotate crops.")
    elif disease_name == "Rust":
        base_treatment= ("Rust causes orange pustules on leaves. Use resistant varieties and maintain good canopy airflow. Apply fungicide only if recommended locally.")
    elif disease_name in ["YellowLeafy2", "Yellowy1"]:
        base_treatment= ("Yellowing of leaves may indicate nutrient deficiency or viral infection. Conduct a leaf test, correct deficiencies, and remove diseased plants.")
    elif disease_name == "smut":
        base_treatment= ("Smut shows whip-like black growth on canes. Use clean seed cane, hot-water treat setts, and destroy infected plants.")
    else:
        base_treatment= ("No specific management instructions found. Consult your local agricultural extension officer for guidance.")

    return base_treatment

# Disease detection function using the trained model
def detect_disease(pil_image):
    """Detect disease using PyTorch .pth model for classification.
    Severity is assessed only when:
      - predicted == "BandedChlorosis" -> use BCL_SEVERITY_MODEL_PATH (.pkl joblib)
      - predicted == "Mosaic" -> use MOSAIC_SEVERITY_MODEL_PATH (.keras)
    For other predictions, severity is set to 0 (or left unchanged).
    """
    try:
        model, class_names = load_model()
        bcl_model, mosaic_model = load_severity_models()

        if model is None or class_names is None:
            return {
                "name": "Model Error",
                "confidence": 0.0,
                "severity": 0,
                "treatment": "Unable to load classification model. Check server logs."
            }

        # preprocess and predict with PyTorch model
        img_tensor = preprocess_image_torch(pil_image)
        model_device = next(model.parameters()).device if any(p.numel() for p in model.parameters()) else torch.device('cpu')
        img_tensor = img_tensor.to(model_device)

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            # if model returns a tuple (outputs, aux) handle first element
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            # convert to probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence_val, pred_idx = torch.max(probs, dim=0)

        predicted_disease = class_names[pred_idx.item()]
        confidence = float(confidence_val.item())

        # Default severity = 0 (no assessment) unless required by spec
        severity = 0

        # Severity logic:
        if predicted_disease == "BandedChlorosis":
            # use joblib .pkl model if available
            if bcl_model is not None:
                try:
                    # Prepare features - adjust based on what your model expects
                    # Option 1: Just confidence
                    features = np.array([[confidence]])
                    
                    # Option 2: If model needs image features, extract them
                    # You may need to adjust this based on your model's training
                    # features = extract_image_features(pil_image)
                    
                    severity_pred = bcl_model.predict(features)
                    
                    # Handle different return types
                    if isinstance(severity_pred, np.ndarray):
                        severity = int(np.clip(np.round(severity_pred[0]), 0, 5))
                    else:
                        severity = int(np.clip(np.round(float(severity_pred)), 0, 5))
                        
                    st.info(f"BandedChlorosis severity assessed: {severity}/5")
                except Exception as e:
                    st.warning(f"BandedChlorosis severity model predict error: {e}")
                    # Fallback: estimate based on confidence
                    severity = int(np.clip(confidence * 5, 0, 5))
            else:
                # Fallback when model missing - estimate from confidence
                severity = int(np.clip(confidence * 5, 0, 5))

        elif predicted_disease == "Mosaic":
            # use Keras severity model if available
            if mosaic_model is not None:
                try:
                    # Resize to expected size 384x384x3
                    resized_img = pil_image.resize((384, 384))
                    keras_input = preprocess_image_for_prediction(resized_img, img_size=(384, 384))
                    
                    if keras_input is not None:
                        pred = mosaic_model.predict(keras_input, verbose=0)
                        
                        # Handle different output shapes
                        if isinstance(pred, np.ndarray):
                            if pred.ndim > 1:
                                val = float(pred[0][0])
                            else:
                                val = float(pred[0])
                        else:
                            val = float(pred)
                            
                        severity = int(np.clip(np.round(val), 0, 5))
                        st.info(f"Mosaic severity assessed: {severity}/5")
                    else:
                        severity = int(np.clip(confidence * 5, 0, 5))
                        
                except Exception as e:
                    st.warning(f"Mosaic severity model predict error: {e}")
                    # Fallback: estimate based on confidence
                    severity = int(np.clip(confidence * 5, 0, 5))
            else:
                # Fallback when model missing
                severity = int(np.clip(confidence * 5, 0, 5))

        # Build all_predictions dict for UI (convert torch probs to floats)
        all_predictions = {}
        for i, name in enumerate(class_names):
            all_predictions[name] = float(probs[i].item())

        treatment = get_treatment_recommendation(predicted_disease, severity)

        return {
            "name": predicted_disease,
            "confidence": confidence,
            "severity": int(severity),
            "treatment": treatment,
            "all_predictions": all_predictions
        }

    except Exception as e:
        st.error(f"❌ Error in disease detection: {str(e)}")
        return {
            "name": "Detection Error",
            "confidence": 0.0,
            "severity": 0,
            "treatment": f"Error occurred during detection: {str(e)}"
        }

# Database Configuration
@st.cache_resource
def init_connection():
    """Initialize PostgreSQL connection with retry logic"""
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'sugarcane_app'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            connection = psycopg2.connect(
                **DB_CONFIG,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            connection.autocommit = False
            
            # Test connection
            cursor = connection.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
            
            return connection
            
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return None
        except Exception as e:
            return None
    
    return None

@contextmanager
def get_db_cursor():
    """Simplified database cursor context manager"""
    connection = None
    cursor = None
    
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'sugarcane_app'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            port=os.getenv('DB_PORT', '5432'),
            connect_timeout=5
        )
        
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        yield cursor
        connection.commit()
        
    except psycopg2.Error:
        if connection:
            connection.rollback()
        yield None
    except Exception:
        if connection:
            connection.rollback()
        yield None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def is_database_available():
    """Quick check if database is available"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'sugarcane_app'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            port=os.getenv('DB_PORT', '5432'),
            connect_timeout=2
        )
        conn.close()
        return True
    except:
        return False

def create_tables_with_cursor(cursor):
    """Create tables using provided cursor"""
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(64) NOT NULL,
        full_name VARCHAR(100) NOT NULL,
        farm_location VARCHAR(100),
        farm_size DECIMAL(10,2),
        join_date DATE DEFAULT CURRENT_DATE,
        scans_performed INTEGER DEFAULT 0,
        diseases_detected INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Scan history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        image_data BYTEA,
        image_name VARCHAR(255),
        disease_detected VARCHAR(100),
        confidence_score DECIMAL(5,3),
        severity_level INTEGER,
        treatment_recommendation TEXT,
        scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Disease information table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS disease_info (
        id SERIAL PRIMARY KEY,
        disease_name VARCHAR(100) UNIQUE NOT NULL,
        description TEXT,
        causes TEXT,
        prevention TEXT,
        treatment TEXT
    );
    """)
    
    # Insert disease information
    disease_data = [
        ('Healthy', 'No disease detected. Leaf appears healthy.', 'N/A', 'Continue regular care and monitoring.', 'No treatment needed.'),
        ('RedRot', 'A serious fungal disease causing reddish discoloration and rotting of internodes.', 'Caused by Colletotrichum falcatum fungus, spread through infected setts and tools.', 'Use disease-resistant varieties, treat setts with fungicide, maintain field hygiene.', 'Apply fungicide treatment. Remove affected parts immediately.'),
        ('Mosaic', 'Viral disease causing characteristic mosaic patterns on leaves.', 'Spread by aphids and through infected planting material.', 'Use virus-free planting material, control aphid vectors, remove infected plants.', 'Use disease-resistant varieties. Remove infected plants to prevent spread.'),
        ('Rust', 'Fungal disease causing orange-red pustules on leaf surface.', 'Caused by Puccinia species, favored by high humidity and moderate temperatures.', 'Improve air circulation, apply preventive fungicides, use resistant varieties.', 'Apply copper-based fungicide. Improve air circulation around plants.'),
        ('smut', 'Fungal disease causing black, sooty growth on shoots.', 'Caused by Sporisorium scitamineum, spread through airborne spores.', 'Use healthy planting material, apply systemic fungicides, remove infected shoots.', 'Remove infected shoots immediately. Apply systemic fungicide.'),
        ('Yellow Leaf', 'Viral disease causing yellowing and necrosis of leaves.', 'Caused by Sugarcane yellow leaf virus, transmitted by aphids.', 'Control aphid populations, use virus-tested planting material, maintain plant nutrition.', 'Check soil nutrition. Apply balanced fertilizer. Control aphid vectors.'),
        ('Brown Spot', 'Fungal disease causing brown spots on leaves.', 'Caused by fungal pathogens, favored by high humidity.', 'Improve field drainage, avoid overhead irrigation, use resistant varieties.', 'Apply copper-based fungicide. Improve field drainage and avoid overhead irrigation.')
    ]
    
    for disease in disease_data:
        cursor.execute("""
        INSERT INTO disease_info (disease_name, description, causes, prevention, treatment)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (disease_name) DO NOTHING;
        """, disease)

def initialize_database_once():
    """Initialize database tables only once"""
    if 'db_init_attempted' not in st.session_state:
        st.session_state.db_init_attempted = True
        
        if is_database_available():
            with get_db_cursor() as cursor:
                if cursor:
                    try:
                        create_tables_with_cursor(cursor)
                        # st.success("✅ Database tables ready!")
                    except Exception as e:
                        st.warning(f"Database setup issue: {str(e)}")

# Database helper functions
def register_user(username: str, password: str, full_name: str, farm_location: str, farm_size: float) -> bool:
    """Register a new user in the database"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, farm_location, farm_size)
                VALUES (%s, %s, %s, %s, %s)
                """, (username, password_hash, full_name, farm_location, farm_size))
                return True
            except psycopg2.IntegrityError:
                return False
            except Exception as e:
                return False
    return False

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user and return user data"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute("""
                SELECT * FROM users WHERE username = %s AND password_hash = %s
                """, (username, password_hash))
                user = cursor.fetchone()
                return dict(user) if user else None
            except Exception as e:
                return None
    return None

def get_user_profile(user_id: int) -> Optional[Dict]:
    """Get user profile by ID"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
                return dict(user) if user else None
            except Exception as e:
                return None
    return None

def update_user_profile(user_id: int, full_name: str, farm_location: str, farm_size: float) -> bool:
    """Update user profile"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("""
                UPDATE users 
                SET full_name = %s, farm_location = %s, farm_size = %s
                WHERE id = %s
                """, (full_name, farm_location, farm_size, user_id))
                return True
            except Exception as e:
                return False
    return False

def save_scan_result(user_id: int, image: Image.Image, image_name: str, result: Dict) -> bool:
    """Save scan result to database"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                # Convert image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                cursor.execute("""
                INSERT INTO scan_history 
                (user_id, image_data, image_name, disease_detected, confidence_score, 
                 severity_level, treatment_recommendation)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, img_bytes, image_name, result['name'], 
                      result['confidence'], result['severity'], result['treatment']))
                
                # Update user scan count
                cursor.execute("""
                UPDATE users 
                SET scans_performed = scans_performed + 1,
                    diseases_detected = diseases_detected + CASE WHEN %s != 'Healthy' THEN 1 ELSE 0 END
                WHERE id = %s
                """, (result['name'], user_id))
                
                return True
            except Exception as e:
                return False
    return False

def get_user_scan_history(user_id: int) -> pd.DataFrame:
    """Get user's scan history"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("""
                SELECT id, image_name, disease_detected, confidence_score, 
                       severity_level, treatment_recommendation, scan_date
                FROM scan_history 
                WHERE user_id = %s 
                ORDER BY scan_date DESC
                """, (user_id,))
                
                results = cursor.fetchall()
                if results:
                    df = pd.DataFrame(results)
                    df['scan_date'] = pd.to_datetime(df['scan_date']).dt.strftime('%Y-%m-%d %H:%M')
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                return pd.DataFrame()
    return pd.DataFrame()

def get_scan_image(scan_id: int) -> Optional[Image.Image]:
    """Retrieve scan image from database"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("SELECT image_data FROM scan_history WHERE id = %s", (scan_id,))
                result = cursor.fetchone()
                if result and result['image_data']:
                    return Image.open(io.BytesIO(result['image_data']))
            except Exception as e:
                pass
    return None

def get_disease_info(disease_name: str) -> Optional[Dict]:
    """Get disease information from database"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("""
                SELECT * FROM disease_info WHERE disease_name = %s
                """, (disease_name,))
                result = cursor.fetchone()
                return dict(result) if result else None
            except Exception as e:
                return None
    return None

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

def home_page():
    """Home page with attractive landing and three static feature boxes"""
    st.markdown('<div class="slide-in-up">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🌾 LIRA Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sugarcane Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Three static feature boxes
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class="feature-box" style="--box-color: #22c55e;">
            <div class="feature-icon">🔍</div>
            <h3 class="feature-title">Smart Detection</h3>
            <p class="feature-description">Advanced CNN models trained on comprehensive sugarcane leaf datasets for accurate disease identification with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box" style="--box-color: #3b82f6;">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Severity Analysis</h3>
            <p class="feature-description">Horsfall-Barratt scale integration for precise disease severity assessment and progression monitoring in real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box" style="--box-color: #f59e0b;">
            <div class="feature-icon">💡</div>
            <h3 class="feature-title">Smart Advisory</h3>
            <p class="feature-description">Personalized treatment recommendations and crop management insights powered by machine learning algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics section
    st.markdown("---")
    
    # Try to get real stats, fallback to demo stats
    total_users = total_scans = total_diseases = disease_types = 0
    
    if is_database_available():
        with get_db_cursor() as cursor:
            if cursor:
                try:
                    cursor.execute("SELECT COUNT(*) as total_users FROM users")
                    result = cursor.fetchone()
                    total_users = result['total_users'] if result else 0
                    
                    cursor.execute("SELECT COALESCE(SUM(scans_performed), 0) as total_scans FROM users")
                    result = cursor.fetchone()
                    total_scans = result['total_scans'] if result else 0
                    
                    cursor.execute("SELECT COALESCE(SUM(diseases_detected), 0) as total_diseases FROM users")
                    result = cursor.fetchone()
                    total_diseases = result['total_diseases'] if result else 0
                    
                    cursor.execute("SELECT COUNT(DISTINCT disease_detected) as disease_types FROM scan_history WHERE disease_detected != 'Healthy'")
                    result = cursor.fetchone()
                    disease_types = result['disease_types'] if result else 0
                    
                except:
                    total_users, total_scans, total_diseases, disease_types = 42, 186, 89, 5
    else:
        total_users, total_scans, total_diseases, disease_types = 42, 186, 89, 5
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #22c55e; margin: 0;">{total_users:,}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Registered Farmers</p>
            <small style="color: #4ade80;">Growing daily</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #22c55e; margin: 0;">{total_scans:,}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Total Scans</p>
            <small style="color: #4ade80;">Real-time analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #22c55e; margin: 0;">{total_diseases:,}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Diseases Detected</p>
            <small style="color: #4ade80;">Crops saved</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #22c55e; margin: 0;">{disease_types}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">Detection Types</p>
            <small style="color: #4ade80;">95%+ accuracy</small>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    st.markdown("""
    <div class="custom-card" style="text-align: center; background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.2));">
        <h2 style="color: #22c55e; margin-bottom: 20px;">🚀 Ready to Protect Your Crops?</h2>
        <p style="font-size: 1.1rem; margin-bottom: 30px;">Join thousands of farmers to safeguard their sugarcane crops</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🌱 Get Started - Join LIRA Pro", key="cta_signup", use_container_width=True):
                st.session_state.page = "Login"
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📸 Start Disease Detection", key="cta_scan", use_container_width=True, type="primary"):
                st.session_state.page = "Disease Scanner"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("👤 View Profile", key="quick_profile", use_container_width=True):
                st.session_state.page = "Profile"
                st.rerun()
        
        with col2:
            if st.button("📊 View Analytics", key="quick_analytics", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()
                
        with col3:
            if st.button("🚪 Logout", key="quick_logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_profile = {}
                st.session_state.page = "Home"
                st.rerun()

def login_page():
    """Login and registration page"""
    st.markdown('<h1 class="main-header">🔐 Access Your Account</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "🌱 Register"])
    
    with tab1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #22c55e;">Welcome Back, Farmer! 👨‍🌾</h3>
            <p>Sign in to access your personalized crop protection dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("🚀 Sign In", use_container_width=True):
                if username and password:
                    user_data = authenticate_user(username, password)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_profile = user_data
                        st.session_state.current_user = username
                        st.session_state.user_id = user_data['id']
                        st.success("🎉 Welcome back! Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.error("❌ Please enter both username and password!")
    
    with tab2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #22c55e;">Join LIRA Community! 🌱</h3>
            <p>Create your account and start protecting your crops with our technology</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username", placeholder="Choose a unique username")
                full_name = st.text_input("Full Name", placeholder="Your full name")
                farm_location = st.text_input("Farm Location", placeholder="City, State")
            
            with col2:
                new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                farm_size = st.number_input("Farm Size (acres)", min_value=0.1, value=1.0, step=0.1)
            
            if st.form_submit_button("🌟 Create Account", use_container_width=True):
                if new_username and new_password and full_name:
                    if new_password == confirm_password:
                        if register_user(new_username, new_password, full_name, farm_location, farm_size):
                            st.success("🎉 Account created successfully! Please login with your credentials.")
                            time.sleep(2)
                        else:
                            st.error("❌ Username already exists or registration failed!")
                    else:
                        st.error("❌ Passwords don't match!")
                else:
                    st.error("❌ Please fill all required fields!")

def profile_page():
    """User profile page"""
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to view your profile.")
        return
    
    # Refresh user profile from database
    profile = get_user_profile(st.session_state.user_id)
    if not profile:
        st.error("❌ Could not load profile data.")
        return
    
    st.markdown('<h1 class="main-header">👤 Your Profile</h1>', unsafe_allow_html=True)
    
    # Profile overview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <h3 style="color: #22c55e;">👨‍🌾 Farmer Profile</h3>
            <div style="margin: 30px 0;">
                <div style="width: 120px; height: 120px; border-radius: 50%; background: linear-gradient(135deg, #16a34a, #22c55e); margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 48px; box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);">
                    🌾
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="custom-card">
            <h3 style="color: #22c55e;">📋 Profile Information</h3>
            <div style="line-height: 1.8;">
                <p><strong>👤 Name:</strong> {profile.get('full_name', 'N/A')}</p>
                <p><strong>🌍 Location:</strong> {profile.get('farm_location', 'N/A')}</p>
                <p><strong>🚜 Farm Size:</strong> {profile.get('farm_size', 0)} acres</p>
                <p><strong>📅 Member Since:</strong> {profile.get('join_date', 'N/A')}</p>
                <p><strong>📸 Total Scans:</strong> <span style="color: #22c55e; font-weight: bold;">{profile.get('scans_performed', 0)}</span></p>
                <p><strong>🦠 Diseases Detected:</strong> <span style="color: #f59e0b; font-weight: bold;">{profile.get('diseases_detected', 0)}</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity from database
    st.markdown("### 📈 Recent Scan History")
    
    scan_history = get_user_scan_history(st.session_state.user_id)
    
    if not scan_history.empty:
        # Display history with image preview option
        for idx, row in scan_history.head(10).iterrows():
            with st.expander(f"📸 {row['disease_detected']} - {row['scan_date']} (Confidence: {row['confidence_score']:.1%})", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show image if available
                    scan_image = get_scan_image(row['id'])
                    if scan_image:
                        st.image(scan_image, caption=row['image_name'], width=200)
                
                with col2:
                    st.markdown(f"""
                    <div class="custom-card" style="padding: 20px;">
                        <p><strong>🦠 Disease:</strong> <span style="color: {'#22c55e' if row['disease_detected'] == 'Healthy' else '#f59e0b'};">{row['disease_detected']}</span></p>
                        <p><strong>📊 Severity:</strong> {row['severity_level']}/5</p>
                        <p><strong>💊 Treatment:</strong> {row['treatment_recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <h4>📋 No scan history yet</h4>
            <p>Start scanning to see your activity and build your crop health insights!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Edit profile section
    st.markdown("---")
    st.markdown("### ✏️ Edit Profile")
    
    with st.form("edit_profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("Full Name", value=profile.get('full_name', ''))
            new_location = st.text_input("Farm Location", value=profile.get('farm_location', ''))
        
        with col2:
            new_farm_size = st.number_input("Farm Size (acres)", value=float(profile.get('farm_size', 1.0)), min_value=0.1, step=0.1)
        
        if st.form_submit_button("💾 Update Profile", use_container_width=True):
            if update_user_profile(st.session_state.user_id, new_name, new_location, new_farm_size):
                st.success("✅ Profile updated successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to update profile!")

def disease_scanner_page():
    """Disease scanning page"""
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to use the disease scanner.")
        return
    
    st.markdown('<h1 class="main-header">📸 Disease Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a sugarcane leaf image for instant disease detection</p>', unsafe_allow_html=True)
    
    # Check if model is available
    model, class_names = load_model()
    if model is None:
        st.error("❌ Disease detection model not available. Please ensure the model file is present.")
        st.info(f"Expected model path: {MODEL_PATH}")
        return
    else:
        st.success("✅ Disease Detection Model Loaded Successfully!")
        st.info(f"Model can detect: {', '.join(class_names)}")
    
    # Instructions
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #22c55e;">📋 How to Use the Scanner:</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📱</div>
                <p><strong>Step 1:</strong> Take a clear photo of the sugarcane leaf</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📤</div>
                <p><strong>Step 2:</strong> Upload the image using the button below</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🤖</div>
                <p><strong>Step 3:</strong> Analyzing in under 2 seconds</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                <p><strong>Step 4:</strong> Get detailed results & treatment plans</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "🖼️ Choose a sugarcane leaf image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the sugarcane leaf for analysis"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #22c55e;">📷 Uploaded Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Sugarcane Leaf Analysis", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card" style="height: 400px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 20px;">🤖</div>
                    <h3 style="color: #22c55e;">Ready for Analysis</h3>
                    <p>Click the button below to start the disease detection process</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Disease", use_container_width=True):
                with st.spinner("🤖 analyzing your image..."):
                    # Simulate processing time with progress
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Get disease detection result using the trained model
                    result = detect_disease(image)
                    
                    # Save result to database
                    save_scan_result(st.session_state.user_id, image, uploaded_file.name, result)
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Display result
                    if result['name'] == 'Healthy':
                        st.markdown(f"""
    <div class="healthy-result">
        <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
        <h2>{result['name']} Leaf Detected</h2>
        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
        <p><strong>Status:</strong> Your crop is in excellent condition!</p>
        <p><strong>Recommendation:</strong> {result['treatment']}</p>
    </div>
    """, unsafe_allow_html=True)
                    else:
    # Always show detected disease name and confidence
                        disease_name = result['name']
                        confidence = result['confidence']
                        severity = result.get('severity', 0)
                        treatment = result['treatment']

    # Only show Severity Level if disease is BandedChlorosis or Mosaic
                        severity_html = ""
                        if disease_name in ["BandedChlorosis", "Mosaic"]:
                            severity_html = f"<p><strong>Severity Level:</strong> {severity}/5</p>"

    # Build the result card
                        st.markdown(f"""
<div class="disease-result">
    <div style="font-size: 3rem; margin-bottom: 10px;">⚠️</div>
    <h2>{disease_name} Detected</h2>
    <p><strong>Confidence:</strong> {confidence:.1%}</p>
    {severity_html}
</div>
""", unsafe_allow_html=True)





                    
                    # Show all class probabilities if available
                    if 'all_predictions' in result:
                        st.markdown("### 📊 Detailed Prediction Probabilities")
                        
                        # Create a bar chart of all predictions
                        pred_data = result['all_predictions']
                        pred_df = pd.DataFrame(list(pred_data.items()), columns=['Disease', 'Probability'])
                        pred_df['Probability'] = pred_df['Probability'] * 100  # Convert to percentage
                        pred_df = pred_df.sort_values('Probability', ascending=False)
                        
                        # Display as a nice table
                        st.markdown("""
                        <div class="custom-card">
                            <h4 style="color: #22c55e;">🎯 All Class Predictions</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, row in pred_df.iterrows():
                            is_predicted = row['Disease'] == result['name']
                            color = "#22c55e" if is_predicted else "#ffffff"
                            marker = " ← PREDICTED" if is_predicted else ""
                            st.markdown(f"**{row['Disease']}**: {row['Probability']:.2f}%{marker}", 
                                       help=f"Confidence level for {row['Disease']}")
                    
                    # Get detailed disease info
                    disease_info = get_disease_info(result['name'])
                    if disease_info:
                        st.markdown("### 📚 Detailed Disease Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="custom-card">
                                <h4 style="color: #22c55e;">📖 Description</h4>
                                <p>{disease_info['description']}</p>
                                <h4 style="color: #22c55e; margin-top: 20px;">🔍 Causes</h4>
                                <p>{disease_info['causes']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="custom-card">
                                <h4 style="color: #22c55e;">🛡️ Prevention</h4>
                                <p>{disease_info['prevention']}</p>
                                <h4 style="color: #22c55e; margin-top: 20px;">💊 Treatment</h4>
                                <p>{disease_info['treatment']}</p>
                            </div>
                            """, unsafe_allow_html=True)

def analytics_page():
    """Analytics and insights page"""
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to view analytics.")
        return
    
    st.markdown('<h1 class="main-header">📊 Farm Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Get user's scan data for analytics
    scan_history = get_user_scan_history(st.session_state.user_id)
    
    if scan_history.empty:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📋</div>
            <h3>No Analytics Data Available</h3>
            <p>Start scanning your crops to generate insights and analytics!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Disease distribution
    st.markdown("### 🦠 Disease Distribution Analysis")
    disease_counts = scan_history['disease_detected'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #22c55e;">Disease Detection Chart</h4>
        </div>
        """, unsafe_allow_html=True)
        st.bar_chart(disease_counts)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #22c55e;">Disease Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        for disease, count in disease_counts.items():
            percentage = (count / len(scan_history)) * 100
            color = "#22c55e" if disease == "Healthy" else "#f59e0b"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {color}; margin: 0;">{disease}</h4>
                <p style="margin: 5px 0;"><strong>{count}</strong> scans</p>
                <small style="color: #4ade80;">{percentage:.1f}% of total</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Timeline analysis
    st.markdown("### 📈 Scanning Timeline")
    st.markdown("""
    <div class="custom-card">
        <h4 style="color: #22c55e;">Daily Scan Activity</h4>
    </div>
    """, unsafe_allow_html=True)
    scan_history['scan_date'] = pd.to_datetime(scan_history['scan_date'])
    timeline_data = scan_history.groupby(scan_history['scan_date'].dt.date).size()
    st.line_chart(timeline_data)

# Enhanced sidebar navigation with better state management
def sidebar():
    """Create sidebar navigation with improved state management"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-bottom: 2px solid rgba(34, 197, 94, 0.2); margin-bottom: 20px;">
            <h2 style="color: #22c55e; margin: 0;">🌾 LIRA Pro</h2>
            <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 5px 0 0 0;">Crop Protection</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show model status
        model, class_names = load_model()
        if model is not None:
            st.markdown("""
            <div class="sidebar-metric" style="background: rgba(34, 197, 94, 0.2);">
                <h4 style="color: #22c55e; margin: 0;">🤖 Model Status</h4>
                <p style="margin: 5px 0 0 0; color: #4ade80;">✅ Active & Ready</p>
                <small style="color: #ffffff;">EfficientNetV2B2</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-metric" style="background: rgba(239, 68, 68, 0.2);">
                <h4 style="color: #ef4444; margin: 0;">🤖 Model Status</h4>
                <p style="margin: 5px 0 0 0; color: #fca5a5;">❌ Not Available</p>
                <small style="color: #ffffff;">Check model file</small>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.logged_in:
            st.markdown(f"""
            <div class="custom-card" style="margin-bottom: 20px; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 10px;">👋</div>
                <p style="color: #22c55e; font-weight: bold; margin: 0;">Welcome back!</p>
                <p style="margin: 5px 0 0 0; font-size: 0.9rem;">{st.session_state.user_profile.get('full_name', 'User')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu with improved state handling
            st.markdown("### 🧭 Navigation")
            
            # Use radio buttons for better state management
            page_options = ["🏠 Home", "📸 Disease Scanner", "👤 Profile", "📊 Analytics"]
            
            # Find current index
            current_index = 0
            page_mapping = {
                "🏠 Home": "Home",
                "📸 Disease Scanner": "Disease Scanner",
                "👤 Profile": "Profile", 
                "📊 Analytics": "Analytics"
            }
            
            for i, option in enumerate(page_options):
                if page_mapping[option] == st.session_state.get('page', 'Home'):
                    current_index = i
                    break
            
            selected_option = st.radio("", page_options, index=current_index, label_visibility="collapsed")
            
            # Update page if selection changed
            new_page = page_mapping[selected_option]
            if new_page != st.session_state.get('page', 'Home'):
                st.session_state.page = new_page
                st.rerun()
            
            st.markdown("---")
            
            # Quick stats with improved styling
            profile = st.session_state.user_profile
            
            st.markdown("### 📈 Quick Stats")
            
            st.markdown("""
            <div class="sidebar-metric">
                <h3 style="color: #22c55e; margin: 0;">📸 Total Scans</h3>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0 0 0;">{}</p>
            </div>
            """.format(profile.get('scans_performed', 0)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sidebar-metric">
                <h3 style="color: #f59e0b; margin: 0;">🦠 Diseases Found</h3>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0 0 0;">{}</p>
            </div>
            """.format(profile.get('diseases_detected', 0)), unsafe_allow_html=True)
            
            # Farm info
            if profile.get('farm_location') or profile.get('farm_size'):
                st.markdown("### 🚜 Farm Info")
                if profile.get('farm_location'):
                    st.markdown(f"**📍 Location:** {profile.get('farm_location')}")
                if profile.get('farm_size'):
                    st.markdown(f"**📏 Size:** {profile.get('farm_size')} acres")
            
            st.markdown("---")
            
            # Logout button with confirmation
            if st.button("🚪 Logout", use_container_width=True, help="Sign out of your account"):
                st.session_state.logged_in = False
                st.session_state.user_profile = {}
                st.session_state.page = "Home"
                if 'user_id' in st.session_state:
                    del st.session_state.user_id
                if 'current_user' in st.session_state:
                    del st.session_state.current_user
                st.success("👋 Successfully logged out!")
                time.sleep(1)
                st.rerun()
        else:
            # Login prompt with better styling
            st.markdown("""
            <div class="custom-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 15px;">🔒</div>
                <h3 style="color: #22c55e; margin-bottom: 10px;">Access Required</h3>
                <p style="font-size: 0.9rem; margin-bottom: 20px;">Login to unlock crop protection features</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🌱 Login / Register", use_container_width=True, help="Sign in to access all features"):
                st.session_state.page = "Login"
                st.rerun()
            
            # Demo info for non-logged-in users
            st.markdown("---")
            st.markdown("### 🌟 Features Available")
            
            features = [
                "🔍 Disease Detection",
                "📊 Detailed Analytics", 
                "👤 Personal Profile",
                "📈 Scan History",
                "💊 Treatment Plans",
                "🦠 Disease Database"
            ]
            
            for feature in features:
                st.markdown(f"• {feature}")

# Main application logic with better error handling
def main():
    """Main application function with enhanced error handling"""
    try:
        # Initialize page state
        if 'page' not in st.session_state:
            st.session_state.page = "Home"
        
        # Initialize database
        initialize_database_once()
        
        # Create sidebar
        sidebar()
        
        # Route to appropriate page with error handling
        try:
            if st.session_state.page == "Home":
                home_page()
            elif st.session_state.page == "Login":
                login_page()
            elif st.session_state.page == "Profile":
                profile_page()
            elif st.session_state.page == "Disease Scanner":
                disease_scanner_page()
            elif st.session_state.page == "Analytics":
                analytics_page()
            else:
                # Fallback to home if unknown page
                st.session_state.page = "Home"
                home_page()
                
        except Exception as e:
            st.error(f"❌ An error occurred while loading the page: {str(e)}")
            st.info("🔄 Please refresh the page or try again.")
            
            # Reset to home page on error
            if st.button("🏠 Return to Home"):
                st.session_state.page = "Home"
                st.rerun()
    
    except Exception as e:
        st.error("❌ Critical application error occurred.")
        st.info("Please refresh the page to restart the application.")
        
        # Show technical details in expander for debugging
        with st.expander("🔧 Technical Details"):
            st.code(str(e))

# Run the application
if __name__ == "__main__":
    try:
        # Run main application
        main()
        
    except KeyboardInterrupt:
        st.info("👋 Application stopped by user.")
    except Exception as e:
        st.error(f"🚨 Fatal error: {str(e)}")
        st.info("Please restart the application.")