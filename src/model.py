# Code and architecture created by Luis Villeda
import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalRiskModel(nn.Module):
    """
    Late-Fusion Multimodal Architecture for Driving Risk Intelligence.
    Combines Vision (ResNet), Structured Data (MLP), and Text (GRU/Embeddings).
    """
    def __init__(self, num_structured_features: int, text_embedding_dim: int, num_classes: int = 5):
        super(MultimodalRiskModel, self).__init__()
        
        # 1. Vision Branch (Pretrained ResNet18)
        # We use a lightweight ResNet for fast inference
        resnet = models.resnet18(pretrained=True)
        # Remove the classification head
        self.vision_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_out_dim = 512 # ResNet18 output dim
        
        # 2. Structured Data Branch (MLP)
        self.structured_branch = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.structured_out_dim = 128
        
        # 3. Text Branch (Simple GRU for incident reports)
        self.text_branch = nn.GRU(
            input_size=text_embedding_dim, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        self.text_out_dim = 128
        
        # 4. Fusion Layer (Gated Linear Unit - GLU style or simple concat + MLP)
        # Total concatenated dimension
        self.fusion_dim = self.vision_out_dim + self.structured_out_dim + self.text_out_dim
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Risk score regressor (0.0 to 1.0)
        self.risk_regressor = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor, structured_data: torch.Tensor, text_embeddings: torch.Tensor):
        """
        Forward pass for the multimodal model.
        Args:
            image: (Batch, C, H, W)
            structured_data: (Batch, Features)
            text_embeddings: (Batch, Seq_Len, Embed_Dim)
        Returns:
            logits: (Batch, Num_Classes)
            risk_score: (Batch, 1)
        """
        # Vision features
        v_feat = self.vision_branch(image)
        v_feat = v_feat.view(v_feat.size(0), -1) # Flatten
        
        # Structured features
        s_feat = self.structured_branch(structured_data)
        
        # Text features
        # Take the last hidden state of the GRU
        _, t_feat = self.text_branch(text_embeddings)
        t_feat = t_feat.squeeze(0) # (Batch, Hidden_Size)
        
        # Late Fusion (Concatenation)
        fused_features = torch.cat((v_feat, s_feat, t_feat), dim=1)
        
        # Outputs
        logits = self.fusion_classifier(fused_features)
        risk_score = self.risk_regressor(fused_features)
        
        return logits, risk_score
