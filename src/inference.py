# Code and architecture created by Luis Villeda
import os
import torch
import torchvision.ops as ops
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from typing import Dict, Any
from PIL import Image, ImageDraw
import io
import base64
from ultralytics import YOLO

load_dotenv()

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class InferenceEngine:
    """Handles model inference, deduplication, rendering, and LLM-based reasoning."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLO26 (falling back to yolov8n if yolo26n is not available locally)
        try:
            self.model = YOLO('yolo26n.pt')
        except:
            self.model = YOLO('yolov8n.pt')
            
        self.conf_threshold = 0.45
        self.iou_threshold = 0.40
        
        # Hierarchical Color Coding
        self.color_map = {
            'ambulance': (239, 68, 68),   # Red 500
            'firetruck': (185, 28, 28),   # Red 700
            'police': (52, 211, 153),     # Emerald 400
            'firefighter': (5, 150, 105), # Emerald 600
        }
        self.default_color = (100, 100, 100)

    def process_image(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. YOLO Detection
        results = self.model(image, verbose=False)[0]
        
        raw_boxes = results.boxes.xyxy.cpu()
        raw_scores = results.boxes.conf.cpu()
        raw_classes = results.boxes.cls.cpu().int()
        
        # 2. Strict Deduplication (NMS + Confidence)
        conf_mask = raw_scores >= self.conf_threshold
        boxes = raw_boxes[conf_mask]
        scores = raw_scores[conf_mask]
        classes = raw_classes[conf_mask]
        
        final_detections = []
        unique_classes = torch.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # Apply NMS per class
            keep_idx = ops.nms(cls_boxes, cls_scores, self.iou_threshold)
            
            for idx in keep_idx:
                cls_id = cls.item()
                label = self.model.names[cls_id]
                final_detections.append({
                    "bbox": cls_boxes[idx].tolist(),
                    "label": label,
                    "color": self.color_map.get(label.lower(), self.default_color)
                })
        
        # 3. Aggregation
        report_summary = {}
        for det in final_detections:
            label = det["label"]
            if label not in report_summary:
                report_summary[label] = {"count": 0, "color": det["color"]}
            report_summary[label]["count"] += 1
            
        # 4. Render Clean Annotations
        annotated_img = self._render_annotations(image, final_detections)
        
        # Convert to base64
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str, report_summary, final_detections

    def _render_annotations(self, image: Image.Image, detections: list) -> Image.Image:
        annotated = image.convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            
            # 2px solid border, 20% opacity fill
            draw.rectangle([x1, y1, x2, y2], fill=color + (51,), outline=color + (255,), width=2)
            
        return Image.alpha_composite(annotated, overlay).convert("RGB")

    def generate_reasoning(self, report_text: str, risk_score: float, report_summary: dict, provider: str = "gemini") -> str:
        """Dual-provider reasoning generation."""
        prompt = f"Given a driving incident report: '{report_text}', a calculated risk score of {risk_score:.2f}, and the following detected objects: {report_summary}, provide a brief, professional 2-sentence explanation of the risk factors and recommended action."
        
        if provider == "gemini" and GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    try:
                        print(f"Gemini 1.5 Pro hit rate limit. Falling back to gemini-1.5-flash...")
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        return response.text.strip()
                    except Exception as e2:
                        print(f"Gemini 1.5 Flash also failed: {e2}. Falling back to OpenAI if available.")
                        provider = "openai" # Fallback
                else:
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

    def run_inference(self, report_text: str, structured_data: dict, use_provider: str = "gemini", image_bytes: bytes = None) -> Dict[str, Any]:
        """End-to-end inference pipeline."""
        
        img_b64 = None
        report_summary = {}
        detections_legend = []
        
        if image_bytes:
            img_b64, report_summary, detections = self.process_image(image_bytes)
            
        # Mock risk score calculation based on detections
        risk_score = 0.2
        hazard_category = 0
        
        if report_summary:
            # Re-implement risk score calculation based on new report_summary structure
            # report_summary is now {label: {"count": count, "color": color}}
            for label, data in report_summary.items():
                count = data["count"]
                if 'person' in label.lower():
                    risk_score += 0.4 * count
                    hazard_category = 2
                elif 'car' in label.lower() or 'truck' in label.lower() or 'bus' in label.lower():
                    risk_score += 0.3 * count
                    hazard_category = 1 if hazard_category == 0 else hazard_category
                
        risk_score = min(risk_score + (structured_data.get("speed_mph", 0) / 100.0) * 0.3, 1.0)
        
        # Generate LLM Reasoning
        reasoning = self.generate_reasoning(
            report_text=report_text, 
            risk_score=risk_score,
            report_summary=report_summary,
            provider=use_provider
        )
        
        return {
            "hazard_category": hazard_category,
            "risk_score": risk_score,
            "reasoning": reasoning,
            "provider_used": use_provider,
            "annotated_image_base64": img_b64,
            "report_summary": report_summary,
            "detections": detections
        }
