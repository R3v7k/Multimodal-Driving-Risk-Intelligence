import os
import json
import torch
import numpy as np
import google.generativeai as genai
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DrivingDataEngine:
    """
    A 'Principal' level data engine that uses Gemini 1.5 Pro to generate 
    procedural synthetic driving incident metadata and labels.
    """
    def __init__(self, api_key=None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def generate_synthetic_scenarios(self, count=10):
        """
        Generates N synthetic driving scenarios in JSON format.
        Each scenario includes: speed, weather, time, text_report, and ground_truth_risk.
        """
        prompt = f"""
        Generate {count} unique autonomous driving risk scenarios.
        Return ONLY a JSON array of objects with these keys:
        - scenario_id: int
        - speed_mph: float (0-80)
        - weather: string (Clear, Rain, Fog, Snow, Ice)
        - time_of_day: string (Day, Night, Dawn, Dusk)
        - incident_report: string (1-2 sentence description of a hazard)
        - risk_score: float (0.0 to 1.0)
        - hazard_class: string (Pedestrian, Obstruction, Construction, Aggressive Driver, None)

        Example:
        {{
            "scenario_id": 1,
            "speed_mph": 45.5,
            "weather": "Rain",
            "time_of_day": "Night",
            "incident_report": "Vehicle ahead hydroplaned and is spinning across three lanes.",
            "risk_score": 0.95,
            "hazard_class": "Obstruction"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean Markdown formatting if present
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Error generating data: {e}")
            return []

class MultimodalDrivingDataset(Dataset):
    """
    PyTorch Dataset that handles the fusion of Images and Synthetic Metadata.
    """
    def __init__(self, scenarios, transform=None):
        self.scenarios = scenarios
        self.transform = transform
        # Mapping for categorical data
        self.weather_map = {"Clear": 0, "Rain": 1, "Fog": 2, "Snow": 3, "Ice": 4}
        self.hazard_map = {"None": 0, "Pedestrian": 1, "Obstruction": 2, "Construction": 3, "Aggressive Driver": 4}

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        s = self.scenarios[idx]
        
        # 1. Structured Data Vector
        # Normalize speed (approx max 100)
        speed_norm = s['speed_mph'] / 100.0
        weather_idx = self.weather_map.get(s['weather'], 0) / 4.0
        
        structured_vec = torch.tensor([speed_norm, weather_idx], dtype=torch.float32)
        
        # 2. Labels
        risk_label = torch.tensor([s['risk_score']], dtype=torch.float32)
        hazard_label = torch.tensor(self.hazard_map.get(s['hazard_class'], 0), dtype=torch.long)
        
        # Note: In a real run, you would load an actual image here.
        # For this 'Build Mode' demo, we return a placeholder if no image path exists.
        placeholder_image = torch.randn(3, 224, 224) 
        
        return {
            "image": placeholder_image,
            "structured": structured_vec,
            "risk": risk_label,
            "hazard": hazard_label,
            "report": s['incident_report']
        }

if __name__ == "__main__":
    # Test Run
    engine = DrivingDataEngine()
    print("🚀 Generating 5 scenarios via Gemini...")
    data = engine.generate_synthetic_scenarios(count=5)
    
    if data:
        dataset = MultimodalDrivingDataset(data)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch in dataloader:
            print(f"Batch Structured Shape: {batch['structured'].shape}")
            print(f"Example Risk Score: {batch['risk'][0].item()}")
            break
