from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import InferenceEngine

app = FastAPI(title="Multimodal Driving Risk Intelligence API")
engine = InferenceEngine()

class InferenceResponse(BaseModel):
    hazard_category: int
    risk_score: float
    reasoning: str
    provider_used: str

@app.get("/")
def read_root():
    return {"status": "online", "service": "Multimodal Driving Risk API"}

@app.post("/predict", response_model=InferenceResponse)
async def predict_risk(
    report_text: str = Form(...),
    speed_mph: float = Form(...),
    weather_condition: int = Form(...),
    time_of_day: int = Form(...),
    driver_alertness: float = Form(...),
    image: UploadFile = File(None),
    provider: str = Form("gemini")
):
    """
    Endpoint to predict driving risk based on multimodal inputs.
    """
    structured_data = {
        "speed_mph": speed_mph,
        "weather_condition": weather_condition,
        "time_of_day": time_of_day,
        "driver_alertness": driver_alertness
    }
    
    # In a real app, process the uploaded image here
    # image_bytes = await image.read() if image else None
    
    result = engine.run_inference(
        report_text=report_text,
        structured_data=structured_data,
        use_provider=provider
    )
    
    return result
