import os
import json
import torch
from torch.utils.data import Dataset
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class SyntheticDataGenerator:
    """Uses Gemini 1.5 Pro to generate synthetic driving incident data."""
    
    @staticmethod
    def generate_data(num_rows: int = 50, output_file: str = "data/synthetic_incidents.json"):
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set. Cannot generate synthetic data.")
            return
            
        print(f"Generating {num_rows} rows of synthetic driving data using Gemini...")
        
        prompt = f"""
        Generate a JSON array containing {num_rows} synthetic driving incident reports.
        Each object must have the following structure:
        {{
            "incident_id": "string",
            "speed_mph": float (0-120),
            "weather_condition": int (0=Clear, 1=Rain, 2=Snow, 3=Fog),
            "time_of_day": int (0-23),
            "driver_alertness_score": float (0.0 to 1.0),
            "report_text": "string (A short 1-2 sentence description of the incident)",
            "hazard_category": int (0=None, 1=Vehicle, 2=Pedestrian, 3=Animal, 4=Infrastructure),
            "risk_score": float (0.0 to 1.0)
        }}
        Ensure realistic correlations (e.g., high speed + snow + low alertness = high risk).
        Output ONLY valid JSON.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            
            # Clean up response text to extract JSON
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            data = json.loads(response_text)
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Successfully generated and saved data to {output_file}")
            return data
            
        except Exception as e:
            print(f"Error generating data: {e}")
            return None

class DrivingIncidentDataset(Dataset):
    """PyTorch Dataset for Multimodal Driving Data."""
    
    def __init__(self, data_file: str, text_embed_dim: int = 50):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
            
        self.text_embed_dim = text_embed_dim
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Mock Image (3, 224, 224) - In a real scenario, load from disk
        image = torch.randn(3, 224, 224)
        
        # 2. Structured Data
        structured = torch.tensor([
            item['speed_mph'],
            item['weather_condition'],
            item['time_of_day'],
            item['driver_alertness_score']
        ], dtype=torch.float32)
        
        # 3. Text Data (Mock embeddings for simplicity)
        # In production, use a tokenizer/embedding layer or pre-computed embeddings
        seq_len = 10
        text_embeds = torch.randn(seq_len, self.text_embed_dim)
        
        # 4. Labels
        hazard_class = torch.tensor(item['hazard_category'], dtype=torch.long)
        risk_score = torch.tensor([item['risk_score']], dtype=torch.float32)
        
        return image, structured, text_embeds, hazard_class, risk_score

if __name__ == "__main__":
    # Generate data if run directly
    SyntheticDataGenerator.generate_data()
