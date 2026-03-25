# Code and architecture created by Luis Villeda
import streamlit as st
import requests
import json
import base64
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(page_title="Driving Risk Intelligence", layout="wide")

st.title("🚗 Multimodal Driving Risk Intelligence")
st.markdown("Upload incident data to assess road hazard categories and risk scores using YOLO26 and Gemini 1.5 Pro.")

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
                
                # 70/30 Split for modern UI
                col_img, col_data = st.columns([2.5, 1.5], gap="large")
                
                with col_img:
                    if result.get("annotated_image_base64"):
                        img_bytes = base64.b64decode(result["annotated_image_base64"])
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, use_column_width=True, caption="Clean Annotated Scene")
                    else:
                        st.info("No image provided for analysis.")
                
                with col_data:
                    st.subheader("Scene Summary")
                    
                    hazard_map = {0: "None", 1: "Vehicle", 2: "Pedestrian", 3: "Animal", 4: "Infrastructure"}
                    hazard_name = hazard_map.get(result['hazard_category'], "Unknown")
                    
                    st.metric("Hazard Category", hazard_name)
                    st.metric("Risk Score", f"{result['risk_score']:.2f}")
                    st.metric("Provider", result['provider_used'].upper())
                    
                    report_summary = result.get("report_summary")
                    if report_summary:
                        st.markdown("---")
                        st.markdown("### 📊 By Class")
                        for super_cat, data in report_summary.items():
                            st.markdown(f"**{super_cat}: {data['total']}**")
                            for sub_cat, count in data['subtypes'].items():
                                st.markdown(f"- {sub_cat}: {count}")
                    
                    detections_legend = result.get("detections_legend")
                    if detections_legend:
                        st.markdown("---")
                        st.markdown("### 🔍 Detection Legend")
                        with st.expander("View Object Details", expanded=True):
                            for det in detections_legend:
                                icon = "🔴" if det["is_primary"] else "🔵"
                                st.markdown(f"{icon} **ID {det['id']}:** {det['label']} - *{det['confidence']:.2f} conf*")
                    
                    st.markdown("---")
                    st.subheader("🧠 AI Reasoning")
                    st.info(result['reasoning'])
                
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend API. Ensure FastAPI is running on port 8000.")
            st.code("uvicorn app.main:app --reload")
else:
    st.info("👈 Enter incident details in the sidebar and click 'Analyze Risk'.")
