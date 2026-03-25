import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MultimodalRiskModel
from data_engine import DrivingIncidentDataset, SyntheticDataGenerator

def train_model(epochs: int = 10, batch_size: int = 8, lr: float = 1e-4):
    """Training loop with Early Stopping and Checkpointing."""
    
    data_path = "data/synthetic_incidents.json"
    if not os.path.exists(data_path):
        print("Data not found. Generating synthetic data...")
        SyntheticDataGenerator.generate_data(num_rows=100, output_file=data_path)
        
    dataset = DrivingIncidentDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = MultimodalRiskModel(
        num_structured_features=4, 
        text_embedding_dim=50, 
        num_classes=5
    ).to(device)
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_risk = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, structured, text, labels_class, labels_risk) in enumerate(dataloader):
            images = images.to(device)
            structured = structured.to(device)
            text = text.to(device)
            labels_class = labels_class.to(device)
            labels_risk = labels_risk.to(device)
            
            optimizer.zero_grad()
            
            logits, risk_preds = model(images, structured, text)
            
            # Combined loss
            loss_c = criterion_class(logits, labels_class)
            loss_r = criterion_risk(risk_preds, labels_risk)
            loss = loss_c + loss_r
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        # Early Stopping & Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("  -> Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
