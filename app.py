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
import os

# Load .env file
load_dotenv()

# Read DB config from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Page configuration
st.set_page_config(
    page_title="AgriScan Pro - Sugarcane Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sexy UI (same as before)
st.markdown("""
<style>
    /* Main background and container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 30px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(31, 38, 135, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        color: white;
        padding: 15px;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Success/error message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.5);
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.2);
        border: 1px solid rgba(244, 67, 54, 0.5);
        border-radius: 10px;
    }
    
    /* Disease result card */
    .disease-result {
        background: linear-gradient(135deg, #FF9A8B 0%, #F5576C 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .healthy-result {
        background: linear-gradient(135deg, #A8E6CF 0%, #4CAF50 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Database Configuration
@st.cache_resource
def init_connection():
    """Initialize PostgreSQL connection"""
    try:
        # Database configuration - update these with your PostgreSQL credentials
        DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'sugarcane_app'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("💡 Please check your PostgreSQL configuration and ensure the database is running.")
        return None

@contextmanager
def get_db_cursor():
    """Context manager for database operations"""
    conn = init_connection()
    if conn:
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            st.error(f"Database error: {str(e)}")
            yield None
        finally:
            cursor.close()
            conn.close()
    else:
        yield None

def create_tables():
    """Create database tables if they don't exist"""
    with get_db_cursor() as cursor:
        if cursor:
            try:
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
                
                # Insert disease information if not exists
                disease_data = [
                    ('Healthy', 'No disease detected. Leaf appears healthy.', 'N/A', 'Continue regular care and monitoring.', 'No treatment needed.'),
                    ('Red Rot', 'A serious fungal disease causing reddish discoloration and rotting of internodes.', 'Caused by Colletotrichum falcatum fungus, spread through infected setts and tools.', 'Use disease-resistant varieties, treat setts with fungicide, maintain field hygiene.', 'Apply fungicide treatment. Remove affected parts immediately.'),
                    ('Mosaic', 'Viral disease causing characteristic mosaic patterns on leaves.', 'Spread by aphids and through infected planting material.', 'Use virus-free planting material, control aphid vectors, remove infected plants.', 'Use disease-resistant varieties. Remove infected plants to prevent spread.'),
                    ('Rust', 'Fungal disease causing orange-red pustules on leaf surface.', 'Caused by Puccinia species, favored by high humidity and moderate temperatures.', 'Improve air circulation, apply preventive fungicides, use resistant varieties.', 'Apply copper-based fungicide. Improve air circulation around plants.'),
                    ('Smut', 'Fungal disease causing black, sooty growth on shoots.', 'Caused by Sporisorium scitamineum, spread through airborne spores.', 'Use healthy planting material, apply systemic fungicides, remove infected shoots.', 'Remove infected shoots immediately. Apply systemic fungicide.'),
                    ('Yellow Leaf', 'Viral disease causing yellowing and necrosis of leaves.', 'Caused by Sugarcane yellow leaf virus, transmitted by aphids.', 'Control aphid populations, use virus-tested planting material, maintain plant nutrition.', 'Check soil nutrition. Apply balanced fertilizer. Control aphid vectors.')
                ]
                
                for disease in disease_data:
                    cursor.execute("""
                    INSERT INTO disease_info (disease_name, description, causes, prevention, treatment)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (disease_name) DO NOTHING;
                    """, disease)
                    
                st.success("✅ Database tables created successfully!")
                
            except Exception as e:
                st.error(f"❌ Error creating tables: {str(e)}")

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
                return False  # Username already exists
            except Exception as e:
                st.error(f"Registration error: {str(e)}")
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
                st.error(f"Authentication error: {str(e)}")
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
                st.error(f"Profile fetch error: {str(e)}")
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
                st.error(f"Profile update error: {str(e)}")
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
                st.error(f"Save scan error: {str(e)}")
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
                st.error(f"History fetch error: {str(e)}")
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
                st.error(f"Image fetch error: {str(e)}")
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
                st.error(f"Disease info error: {str(e)}")
                return None
    return None

# Initialize database tables
if 'db_initialized' not in st.session_state:
    create_tables()
    st.session_state.db_initialized = True

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Mock disease detection function
def detect_disease(image):
    """Mock disease detection - replace with actual ML model"""
    diseases = [
        {"name": "Healthy", "confidence": 0.95, "severity": 0, "treatment": "No treatment needed. Continue regular care."},
        {"name": "Red Rot", "confidence": 0.89, "severity": 3, "treatment": "Apply fungicide treatment. Remove affected parts."},
        {"name": "Mosaic", "confidence": 0.87, "severity": 2, "treatment": "Use disease-resistant varieties. Remove infected plants."},
        {"name": "Rust", "confidence": 0.92, "severity": 4, "treatment": "Apply copper-based fungicide. Improve air circulation."},
        {"name": "Smut", "confidence": 0.85, "severity": 3, "treatment": "Remove infected shoots. Apply systemic fungicide."},
        {"name": "Yellow Leaf", "confidence": 0.78, "severity": 2, "treatment": "Check soil nutrition. Apply balanced fertilizer."}
    ]
    
    # Simulate random prediction
    import random
    result = random.choice(diseases)
    return result

def home_page():
    """Home page with attractive landing"""
    st.markdown('<h1 class="main-header">🌾 AgriScan Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Sugarcane Disease Detection System</p>', unsafe_allow_html=True)
    
    # Feature cards in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>🔍 Smart Detection</h3>
            <p>Advanced CNN models trained on comprehensive sugarcane leaf datasets for accurate disease identification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>📊 Severity Analysis</h3>
            <p>Horsfall-Barratt scale integration for precise disease severity assessment and progression monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card">
            <h3>💡 Smart Advisory</h3>
            <p>Personalized treatment recommendations and crop management insights powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics section with real data from database
    st.markdown("---")
    
    # Get database stats
    with get_db_cursor() as cursor:
        if cursor:
            try:
                cursor.execute("SELECT COUNT(*) as total_users FROM users")
                total_users = cursor.fetchone()['total_users']
                
                cursor.execute("SELECT SUM(scans_performed) as total_scans FROM users")
                total_scans = cursor.fetchone()['total_scans'] or 0
                
                cursor.execute("SELECT SUM(diseases_detected) as total_diseases FROM users")
                total_diseases = cursor.fetchone()['total_diseases'] or 0
                
                cursor.execute("SELECT COUNT(DISTINCT disease_detected) as disease_types FROM scan_history WHERE disease_detected != 'Healthy'")
                disease_types = cursor.fetchone()['disease_types'] or 0
                
            except:
                total_users, total_scans, total_diseases, disease_types = 0, 0, 0, 0
        else:
            total_users, total_scans, total_diseases, disease_types = 0, 0, 0, 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Registered Farmers", f"{total_users:,}", "Growing")
    with col2:
        st.metric("Total Scans", f"{total_scans:,}", "Real-time")
    with col3:
        st.metric("Diseases Detected", f"{total_diseases:,}", "Prevented")
    with col4:
        st.metric("Detection Types", f"{disease_types}", "Accurate")
    
    # CTA Section
    st.markdown("---")
    st.markdown('<h2 style="text-align: center; color: #FFD700;">Ready to Protect Your Crops?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.logged_in:
            if st.button("🚀 Get Started - Sign Up Now", key="cta_signup"):
                st.session_state.page = "Login"
                st.rerun()
        else:
            if st.button("📸 Start Disease Detection", key="cta_scan"):
                st.session_state.page = "Disease Scanner"
                st.rerun()

def login_page():
    """Login and registration page"""
    st.markdown('<h1 class="main-header">🔐 Access Your Account</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.markdown("""
        <div class="custom-card">
            <h3>Welcome Back, Farmer! 👨‍🌾</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("🚀 Login", use_container_width=True):
                if username and password:
                    user_data = authenticate_user(username, password)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_profile = user_data
                        st.session_state.current_user = username
                        st.session_state.user_id = user_data['id']
                        st.success("🎉 Login successful! Welcome back!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.error("❌ Please enter both username and password!")
    
    with tab2:
        st.markdown("""
        <div class="custom-card">
            <h3>Join AgriScan Community! 🌱</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username", placeholder="Choose a username")
                full_name = st.text_input("Full Name", placeholder="Your full name")
                farm_location = st.text_input("Farm Location", placeholder="City, State")
            
            with col2:
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                farm_size = st.number_input("Farm Size (acres)", min_value=0.1, value=1.0, step=0.1)
            
            if st.form_submit_button("🌟 Create Account", use_container_width=True):
                if new_username and new_password and full_name:
                    if new_password == confirm_password:
                        if register_user(new_username, new_password, full_name, farm_location, farm_size):
                            st.success("🎉 Account created successfully! Please login.")
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
        <div class="custom-card">
            <h3>👨‍🌾 Farmer Profile</h3>
            <div style="text-align: center; margin: 20px 0;">
                <div style="width: 120px; height: 120px; border-radius: 50%; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 48px;">
                    🌾
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="custom-card">
            <h3>📋 Profile Information</h3>
            <p><strong>👤 Name:</strong> {profile.get('full_name', 'N/A')}</p>
            <p><strong>🌍 Location:</strong> {profile.get('farm_location', 'N/A')}</p>
            <p><strong>🚜 Farm Size:</strong> {profile.get('farm_size', 0)} acres</p>
            <p><strong>📅 Member Since:</strong> {profile.get('join_date', 'N/A')}</p>
            <p><strong>🔍 Total Scans:</strong> {profile.get('scans_performed', 0)}</p>
            <p><strong>🦠 Diseases Detected:</strong> {profile.get('diseases_detected', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity from database
    st.markdown("### 📈 Recent Scan History")
    
    scan_history = get_user_scan_history(st.session_state.user_id)
    
    if not scan_history.empty:
        # Display history with image preview option
        for idx, row in scan_history.head(10).iterrows():
            with st.expander(f"🔍 {row['disease_detected']} - {row['scan_date']} (Confidence: {row['confidence_score']:.1%})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show image if available
                    scan_image = get_scan_image(row['id'])
                    if scan_image:
                        st.image(scan_image, caption=row['image_name'], width=200)
                
                with col2:
                    st.write(f"**🦠 Disease:** {row['disease_detected']}")
                    st.write(f"**📊 Severity:** {row['severity_level']}/5")
                    st.write(f"**💊 Treatment:** {row['treatment_recommendation']}")
    else:
        st.info("📋 No scan history yet. Start scanning to see your activity!")
    
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
                st.rerun()
            else:
                st.error("❌ Failed to update profile!")

def disease_scanner_page():
    """Disease scanning page"""
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to use the disease scanner.")
        return
    
    st.markdown('<h1 class="main-header">📸 Disease Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a sugarcane leaf image for AI-powered disease detection</p>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="custom-card">
        <h3>📋 How to Use:</h3>
        <p>1. 📱 Take a clear photo of the sugarcane leaf</p>
        <p>2. 📤 Upload the image using the button below</p>
        <p>3. ⏳ Wait for AI analysis (< 2 seconds)</p>
        <p>4. 📊 View detailed results and treatment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "🖼️ Choose a sugarcane leaf image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the sugarcane leaf for analysis"
    )