import streamlit as st
import requests
import json

# Configure Streamlit page
st.set_page_config(page_title="Driving Risk Intelligence", layout="wide")

st.title("🚗 Multimodal Driving Risk Intelligence")
st.markdown("Upload incident data to assess road hazard categories and risk scores using PyTorch and Gemini 1.5 Pro.")

# Sidebar for inputs
with st.sidebar:
    st.header("Incident Data Input")
    
    image_file = st.file_uploader("Upload Dashcam Image", type=["jpg", "png", "jpeg"])
    
    speed = st.slider("Speed (mph)", 0, 120, 45)
    weather = st.selectbox("Weather Condition", options=[0, 1, 2, 3], format_func=lambda x: ["Clear", "Rain", "Snow", "Fog"][x])
    time_of_day = st.slider("Time of Day (Hour)", 0, 23, 14)
    alertness = st.slider("Driver Alertness Score", 0.0, 1.0, 0.8)
    
    report_text = st.text_area("Incident Report", "Sudden braking detected while approaching an intersection.")
    
    provider = st.radio("LLM Provider", ["gemini", "openai"])
    
    analyze_btn = st.button("Analyze Risk", type="primary")

# Main content area
if analyze_btn:
    with st.spinner("Analyzing multimodal data..."):
        # Prepare data for API
        # Assuming FastAPI is running locally on port 8000
        api_url = "http://localhost:8000/predict"
        
        data = {
            "report_text": report_text,
            "speed_mph": speed,
            "weather_condition": weather,
            "time_of_day": time_of_day,
            "driver_alertness": alertness,
            "provider": provider
        }
        
        files = {}
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
            
        try:
            if files:
                response = requests.post(api_url, data=data, files=files)
            else:
                response = requests.post(api_url, data=data)
                
            if response.status_code == 200:
                result = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                hazard_map = {0: "None", 1: "Vehicle", 2: "Pedestrian", 3: "Animal", 4: "Infrastructure"}
                hazard_name = hazard_map.get(result['hazard_category'], "Unknown")
                
                col1.metric("Hazard Category", hazard_name)
                col2.metric("Risk Score", f"{result['risk_score']:.2f}")
                col3.metric("Provider", result['provider_used'].upper())
                
                st.subheader("AI Reasoning")
                st.info(result['reasoning'])
                
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend API. Ensure FastAPI is running on port 8000.")
            st.code("uvicorn app.main:app --reload")
else:
    st.info("👈 Enter incident details in the sidebar and click 'Analyze Risk'.")
