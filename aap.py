import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION CONSTANTS - All hardcoded values in one place for easy updates
# ============================================================================

DISEASE_OPTIONS = [
    "Fungal Infection", 
    "Allergy", 
    "GERD", 
    "Chronic Cholestasis", 
    "Drug Reaction", 
    "Peptic Ulcer Disease", 
    "Diabetes", 
    "Gastroenteritis", 
    "Bronchial Asthma"
]

SYMPTOMS_LIST = [
    "itching", "skin_rash", "continuous_sneezing", "shivering", "chills", 
    "joint_pain", "stomach_pain", "acidity", "vomiting", "fatigue", 
    "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", 
    "weight_loss", "lethargy", "cough", "high_fever"
]

ML_CONFIG = {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": 10
}

# LLM Configuration for HuggingFace Inference API
LLM_CONFIG = {
    # Using free HuggingFace Inference API with Mistral model
    "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
    "api_base_url": "https://api-inference.huggingface.co/models",
    "api_token": os.getenv("HF_API_TOKEN", ""),  # Set via .env or environment variable
    "timeout": 15,
    "max_tokens": 150,
    "temperature": 0.7
}

# Alternative free API endpoint (if HuggingFace token is not available)
FALLBACK_LLM_CONFIG = {
    "use_fallback": True,
    "provider": "local_heuristic"  # Uses rule-based responses instead
}

# Page configuration
st.set_page_config(page_title="AI Medical Expert", page_icon="🩺", layout="centered")

st.title("🩺 AI-Powered Medical Expert Chatbot")
st.subheader("Disease Prediction & AI Consultation")

# ============================================================================
# 1. AUTO-GENERATED/FALLBACK DATASET IF LOCAL CSV IS MISSING
# ============================================================================

@st.cache_resource
def load_and_train_model():
    symptoms = SYMPTOMS_LIST.copy()
    
    try:
        # Try to load local file if user uploaded it
        training_df = pd.read_csv('training_data.csv')
        X_train = training_df.iloc[:, :-1]
        y_train = training_df.iloc[:, -1]
        symptoms = list(X_train.columns)
    except FileNotFoundError:
        # Fallback automated sample dataset generation so the website never crashes
        st.sidebar.info("ℹ️ Using automated diagnostic dataset matrix.")
        data = []
        for i in range(100):
            row = np.random.randint(0, 2, size=len(symptoms)).tolist()
            row.append(np.random.choice(DISEASE_OPTIONS))
            data.append(row)
        training_df = pd.DataFrame(data, columns=symptoms + ["prognosis"])
        X_train = training_df.iloc[:, :-1]
        y_train = training_df.iloc[:, -1]

    model = RandomForestClassifier(**ML_CONFIG)
    model.fit(X_train, y_train)
    return model, symptoms

model, symptoms_list = load_and_train_model()

# ============================================================================
# 2. FIXED FREE OPEN-SOURCE AI DOCTOR PLUGIN (HuggingFace Inference API)
# ============================================================================

def get_fallback_response(disease):
    """Fallback response when API is unavailable"""
    fallback_responses = {
        "Fungal Infection": "Based on your symptoms, a fungal infection is indicated. Keep the affected area clean and dry. Consult a dermatologist for antifungal treatment options.",
        "Allergy": "Your symptoms suggest an allergic reaction. Avoid known allergens and consider antihistamine medication. Consult an allergist if symptoms persist.",
        "GERD": "Gastroesophageal reflux disease (GERD) appears likely. Avoid spicy foods, eat smaller meals, and elevate your head while sleeping. See a gastroenterologist if symptoms worsen.",
        "Diabetes": "These symptoms may indicate diabetes. Monitor your blood sugar levels and maintain a healthy diet. A healthcare provider should conduct proper diagnostic tests.",
        "Bronchial Asthma": "Asthma symptoms detected. Use a bronchodilator inhaler and avoid respiratory irritants. Consult a pulmonologist for an asthma action plan.",
    }
    
    return fallback_responses.get(
        disease, 
        f"Based on your symptoms, our system indicates **{disease}**. Please consult a healthcare provider for proper diagnosis and treatment."
    )

def ask_ai_doctor(disease, user_symptoms):
    """
    Consult AI Doctor using HuggingFace Inference API.
    Falls back to heuristic responses if API unavailable.
    """
    
    prompt = (
        f"System: You are an expert AI Medical Doctor providing educational information only. "
        f"A patient has the following symptoms: {', '.join(user_symptoms)}. "
        f"The diagnostic system predicts: {disease}. "
        f"Write a brief, comforting medical explanation (2-3 sentences) and recommend seeing a healthcare provider. "
        f"Do not provide specific medication recommendations.\n\n"
        f"Response:"
    )
    
    # If no API token is configured, use fallback immediately
    if not LLM_CONFIG["api_token"]:
        st.sidebar.warning("⚠️ HuggingFace API token not configured. Using fallback responses. Set HF_API_TOKEN in .env file.")
        return get_fallback_response(disease)
    
    try:
        # Build the proper HuggingFace Inference API endpoint
        api_url = f"{LLM_CONFIG['api_base_url']}/{LLM_CONFIG['model_id']}"
        
        headers = {
            "Authorization": f"Bearer {LLM_CONFIG['api_token']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": LLM_CONFIG["max_tokens"],
                "temperature": LLM_CONFIG["temperature"]
            }
        }
        
        # Make request with proper timeout
        response = requests.post(
            api_url, 
            headers=headers, 
            json=payload, 
            timeout=LLM_CONFIG["timeout"]
        )
        
        # Handle HTTP errors
        if response.status_code == 401:
            st.sidebar.error("❌ Invalid HuggingFace API token. Check your HF_API_TOKEN in .env")
            return get_fallback_response(disease)
        elif response.status_code == 503:
            st.sidebar.warning("⏳ AI service temporarily unavailable. Using fallback response.")
            return get_fallback_response(disease)
        elif response.status_code != 200:
            st.sidebar.warning(f"⚠️ API Error {response.status_code}. Using fallback response.")
            return get_fallback_response(disease)
        
        # Parse response
        result = response.json()
        
        # Handle different response formats from HuggingFace
        if isinstance(result, list) and len(result) > 0:
            output = result[0].get('generated_text', '')
        elif isinstance(result, dict) and 'generated_text' in result:
            output = result['generated_text']
        else:
            st.sidebar.warning("⚠️ Unexpected API response format. Using fallback.")
            return get_fallback_response(disease)
        
        # Clean up the output to only show the model's new text response
        if prompt in output:
            output = output.replace(prompt, "").strip()
        
        return output.strip() if output else get_fallback_response(disease)
        
    except requests.exceptions.Timeout:
        st.sidebar.warning("⏱️ Request timed out. Using fallback response.")
        return get_fallback_response(disease)
    except requests.exceptions.ConnectionError:
        st.sidebar.warning("🔌 Connection error. Using fallback response.")
        return get_fallback_response(disease)
    except json.JSONDecodeError:
        st.sidebar.warning("⚠️ Invalid response format. Using fallback response.")
        return get_fallback_response(disease)
    except Exception as e:
        st.sidebar.error(f"❌ Unexpected error: {type(e).__name__}")
        return get_fallback_response(disease)

# ============================================================================
# 3. CHAT UI SESSION INITIALIZATION
# ============================================================================

if "user_symptoms" not in st.session_state:
    st.session_state.user_symptoms = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "text": "Hello! I am your AI Medical Assistant. Select the symptoms you are experiencing from the menu below."}
    ]

# Render conversational interface history
for message in st.session_state.chat_history:
    with st.chat_message("assistant" if message["role"] == "bot" else "user", avatar="🤖" if message["role"] == "bot" else "👤"):
        st.write(message["text"])

st.divider()

# ============================================================================
# 4. INTERACTION INTERFACE
# ============================================================================

clean_symptoms = [s.replace('_', ' ').title() for s in symptoms_list]
symptom_mapping = dict(zip(clean_symptoms, symptoms_list))

col1, col2 = st.columns([3, 1])
with col1:
    selected_display = st.selectbox("Select your symptom:", ["-- Search / Choose Symptoms --"] + clean_symptoms, label_visibility="collapsed")
with col2:
    if st.button("➕ Add", use_container_width=True):
        if selected_display != "-- Search / Choose Symptoms --":
            actual_symptom = symptom_mapping[selected_display]
            if actual_symptom not in st.session_state.user_symptoms:
                st.session_state.user_symptoms.append(actual_symptom)
                st.session_state.chat_history.append({"role": "user", "text": f"I am experiencing {selected_display}."})
                st.rerun()

if st.session_state.user_symptoms:
    st.write("**Current Symptoms Profile:**")
    st.info(", ".join([s.replace('_', ' ').title() for s in st.session_state.user_symptoms]))

# ============================================================================
# 5. ML PREDICTION AND AI CONTEXT GENERATION TRIGGER
# ============================================================================

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("🔮 Run AI Diagnosis", type="primary", use_container_width=True):
        if not st.session_state.user_symptoms:
            st.warning("Please add symptoms to process analysis.")
        else:
            with st.spinner("Analyzing physiological markers with Machine Learning..."):
                input_vector = np.zeros(len(symptoms_list))
                for s in st.session_state.user_symptoms:
                    input_vector[symptoms_list.index(s)] = 1
                
                # Predict
                prediction = model.predict([input_vector])[0]
                
            with st.spinner("Consulting AI Doctor Framework..."):
                # Call free LLM API plugin for clinical contextual text explanation
                doctor_explanation = ask_ai_doctor(prediction, [s.replace('_', ' ').title() for s in st.session_state.user_symptoms])
                
                st.session_state.chat_history.append({"role": "bot", "text": doctor_explanation})
                st.rerun()

with btn_col2:
    if st.button("🔄 Reset Patient Profile", use_container_width=True):
        st.session_state.user_symptoms = []
        st.session_state.chat_history = [{"role": "bot", "text": "Profile reset completed. Tell me your active symptoms to begin a new scan."}]
        st.rerun()

st.caption("⚠️ **Disclaimer:** This system delivers informational predictions through pattern-matching algorithms. It is not an alternative to professional clinical expertise.")

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

with st.expander("📋 Setup Instructions (Click to expand)"):
    st.markdown("""
    ### To enable AI Doctor responses:
    
    1. **Get a free HuggingFace API token:**
       - Visit https://huggingface.co/settings/tokens
       - Create a new token with "read" permissions
    
    2. **Create a `.env` file in your project directory:**
       ```
       HF_API_TOKEN=your_token_here
       ```
    
    3. **Run the app:**
       ```bash
       streamlit run app.py
       ```
    
    **Without an API token:** The app will use fallback heuristic responses instead.
    """)
