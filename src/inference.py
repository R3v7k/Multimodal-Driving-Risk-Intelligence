import os
import torch
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class InferenceEngine:
    """Handles model inference and LLM-based reasoning."""
    
    def __init__(self, model_path: str = None):
        # In a real scenario, load the PyTorch model here
        # self.model = MultimodalRiskModel(...)
        # if model_path and os.path.exists(model_path):
        #     self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict_risk_pytorch(self, image_tensor, structured_data, text_data) -> Dict[str, Any]:
        """Mock PyTorch inference."""
        # logits, risk = self.model(image_tensor, structured_data, text_data)
        return {
            "hazard_category": 2, # Mock Pedestrian
            "risk_score": 0.85
        }

    def generate_reasoning(self, report_text: str, risk_score: float, provider: str = "gemini") -> str:
        """Dual-provider reasoning generation."""
        prompt = f"Given a driving incident report: '{report_text}' and a calculated risk score of {risk_score:.2f}, provide a brief, professional 2-sentence explanation of the risk factors and recommended action."
        
        if provider == "gemini" and GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini API failed: {e}. Falling back to OpenAI if available.")
                provider = "openai" # Fallback
                
        if provider == "openai" and OPENAI_API_KEY:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"OpenAI API failed: {e}"
                
        return "Reasoning unavailable. Please check API keys."

    def run_inference(self, report_text: str, structured_data: dict, use_provider: str = "gemini") -> Dict[str, Any]:
        """End-to-end inference pipeline."""
        
        # 1. Run PyTorch Model (Mocked)
        model_results = self.predict_risk_pytorch(None, None, None)
        
        # 2. Generate LLM Reasoning
        reasoning = self.generate_reasoning(
            report_text=report_text, 
            risk_score=model_results["risk_score"],
            provider=use_provider
        )
        
        return {
            "hazard_category": model_results["hazard_category"],
            "risk_score": model_results["risk_score"],
            "reasoning": reasoning,
            "provider_used": use_provider
        }
