# Multimodal Driving Risk Intelligence

A production-grade, end-to-end system for predicting road hazard categories and risk scores using a Late-Fusion Multimodal Architecture.

## Key Features & Recent Updates

1. **React-Based Dashboard (`src/App.tsx`)**: A modern, interactive frontend built with React and Tailwind CSS to visualize the risk intelligence system directly in the browser.
2. **Functional Image Upload**: The dashboard supports uploading dashcam images (.jpg, .png) to simulate visual input for the multimodal engine.
3. **Dynamic LLM Provider Selection**: Users can seamlessly switch between multiple LLM providers and their specific models for risk reasoning:
   - **Gemini** (`gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.5-flash`, `gemini-3.1-pro-preview`)
   - **OpenAI** (`gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`)
   - **Claude** (`claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`)
   - **DeepSeek** (`deepseek-chat`, `deepseek-coder`)
4. **Principal-Level Data Engine (`src/data_engine.py`)**: Uses Gemini 1.5 Pro to procedurally generate synthetic driving incident metadata and labels. Includes a custom PyTorch `MultimodalDrivingDataset` to handle the fusion of images and synthetic metadata.
5. **Updated Training Loop (`src/train.py`)**: Fully integrated with the new `DrivingDataEngine` to automatically generate data and train the PyTorch model with early stopping and checkpointing.

## High-Level System Architecture

The system ingests three modalities of data:
1. **Vision (Images):** Dashcam frames processed via a pretrained ResNet/EfficientNet backbone.
2. **Structured Data:** Telemetry (speed, weather, time) processed via a Multi-Layer Perceptron (MLP).
3. **Text (NLP):** Incident reports processed via a GRU (or Transformer embeddings).

These representations are fused and passed to classification and regression heads. An LLM provides explainable reasoning for the predicted risk.

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
├── app/
│   ├── main.py          # FastAPI backend
│   └── ui.py            # Streamlit dashboard (Python alternative)
├── data/                # Synthetic data storage
├── src/
│   ├── App.tsx          # React Dashboard UI (Browser Preview)
│   ├── data_engine.py   # Gemini synthetic data generation & PyTorch Dataset
│   ├── inference.py     # Multi-provider LLM reasoning & PyTorch inference
│   ├── model.py         # PyTorch Multimodal Architecture
│   └── train.py         # Training loop with early stopping
├── .env.example         # API Keys template
├── requirements.txt     # Python Dependencies
├── package.json         # Node.js Dependencies (React UI)
└── README.md            # Documentation
```

## Quick-Start Guide

### 1. Running the React Dashboard (Node.js)
To run the web-based React dashboard locally:
```bash
npm install
npm run dev
```

### 2. Running the Python Backend & PyTorch Models
To run the full machine learning stack locally:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   ```

3. **Generate Synthetic Data & Train:**
   ```bash
   python src/train.py
   ```

4. **Start the FastAPI Backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Start the Streamlit UI (Alternative Python UI):**
   ```bash
   streamlit run app/ui.py
   ```
