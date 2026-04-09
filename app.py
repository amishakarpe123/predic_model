import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Predictor", page_icon="🤖", layout="centered")

# --- CUSTOM CSS (Animations & Styling) ---
st.markdown("""
    <style>
    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main {
        animation: fadeIn 1.5s ease-out;
    }
    
    /* Glassmorphism effect for the input area */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Customizing the button */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the file is in the same directory.")
        return None

model = load_model()

# --- HEADER ---
st.title("🤖 Intelligent Prediction Portal")
st.write("Enter the parameters below to get a real-time AI prediction.")
st.divider()

# --- INPUT SECTION ---
# Note: Adjust these inputs based on your actual model features
col1, col2 = st.columns(2)

with col1:
    val1 = st.number_input("Feature 1", value=0.0)
    val2 = st.number_input("Feature 2", value=0.0)

with col2:
    val3 = st.number_input("Feature 3", value=0.0)
    val4 = st.number_input("Feature 4", value=0.0)

# --- PREDICTION LOGIC ---
if st.button("✨ Calculate Prediction"):
    if model:
        # Reshape input for the model
        input_data = np.array([[val1, val2, val3, val4]])
        prediction = model.predict(input_data)
        
        # Display Result with a "Success" animation
        st.balloons()
        st.success("Analysis Complete!")
        
        st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #262730;">
                <h2 style="color: #4CAF50;">Prediction Result</h2>
                <h1 style="font-size: 3em;">{prediction[0]}</h1>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model not loaded.")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Streamlit & Scikit-Learn")
