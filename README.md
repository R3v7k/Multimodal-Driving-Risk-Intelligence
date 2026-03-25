# Multimodal Driving Risk Intelligence

A production-grade, end-to-end system for predicting road hazard categories and risk scores using a Late-Fusion Multimodal Architecture.

## High-Level System Architecture

The system ingests three modalities of data:
1. **Vision (Images):** Dashcam frames processed via a pretrained ResNet/EfficientNet backbone.
2. **Structured Data:** Telemetry (speed, weather, time) processed via a Multi-Layer Perceptron (MLP).
3. **Text (NLP):** Incident reports processed via a GRU (or Transformer embeddings).

These representations are fused and passed to classification and regression heads. An LLM (Gemini 1.5 Pro) provides explainable reasoning for the predicted risk.

```text
[Image] ----> [ResNet18] ----\
                              \
[Telemetry] -> [MLP] ---------> [Concat/GLU Fusion] -> [Risk Score & Hazard Class]
                              /
[Text] ------> [GRU] --------/
```

## Mathematical Explanation of the Fusion Layer

Let $v \in \mathbb{R}^{d_v}$ be the vision feature vector, $s \in \mathbb{R}^{d_s}$ be the structured feature vector, and $t \in \mathbb{R}^{d_t}$ be the text feature vector.

In a simple concatenation fusion (as implemented):
$$ f = [v; s; t] \in \mathbb{R}^{d_v + d_s + d_t} $$

The fused vector $f$ is passed through a Multi-Layer Perceptron (MLP) to generate the final logits $L$ and risk score $R$:
$$ L = \text{MLP}_{class}(f) $$
$$ R = \sigma(\text{MLP}_{risk}(f)) $$

Where $\sigma$ is the Sigmoid activation function, ensuring $R \in [0, 1]$.

## Project Structure

```text
multimodal-driving-risk-intelligence/
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py          # FastAPI backend
â”‚   â””â”€â”€ ui.py            # Streamlit dashboard
â”œâ”€â”€ data/                # Synthetic data storage
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_engine.py   # Gemini synthetic data generation & PyTorch Dataset
â”‚   â”œâ”€â”€ inference.py     # Dual-provider LLM reasoning & PyTorch inference
â”‚   â”œâ”€â”€ model.py         # PyTorch Multimodal Architecture
â”‚   â””â”€â”€ train.py         # Training loop with early stopping
â”œâ”€â”€ .env                 # API Keys
â”œâ”€â”€ requirements.txt     # Dependencies
â””â”€â”€ README.md            # Documentation
```

## Quick-Start Guide (Local & Free)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # OPENAI_API_KEY=your_openai_key_here (Optional fallback)
   ```

3. **Generate Synthetic Data & Train (Optional):**
   ```bash
   python src/train.py
   ```

4. **Start the FastAPI Backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Start the Streamlit UI:**
   In a new terminal:
   ```bash
   streamlit run app/ui.py
   ```

Navigate to `http://localhost:8501` to view the dashboard.
