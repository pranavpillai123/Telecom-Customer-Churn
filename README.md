# Telecom Customer Churn Intelligence System

A production-grade Telecom Customer Churn Prediction and Retention Analytics platform powered by an XGBoost classification pipeline, FastAPI backend, and a Google Stitch-inspired Neo-Brutalist dashboard.

---

## Overview

Customer churn poses a major revenue risk for telecommunication providers. This system enables data-driven retention operations by combining:
- A high-performing machine learning inference pipeline (94.6% accuracy, 0.81 F1-score).
- Real-time customer risk scoring and automated retention intervention recommendations.
- Interactive portfolio analytics: risk tier distributions, customer service call cliff analysis, and high-risk customer triage watchlists.
- Counterfactual feature simulation for live customer scenario testing.

---

## System Architecture

- **Machine Learning Pipeline**: Trained XGBoost classifier with scikit-learn preprocessing (StandardScaler for numerical features, OneHotEncoder for state geography, dropping collinear charge fields).
- **Backend**: FastAPI asynchronous REST service providing telemetry analytics and sub-20ms model inference.
- **Frontend**: High-contrast, responsive single-page application styled with Tailwind CSS, Space Grotesk and Inter typography, Google Material Symbols, dynamic SVG gauges, and a dark/light mode toggle.
- **Unified Launcher**: One-click Windows batch runner (`run.bat`) that provisions the environment and serves both API and UI concurrently.

---

## Directory Structure

```
Telecom Churn Prediction/
|-- .antigravity/
|   `-- mcp_config.json          # MCP server configuration
|-- .venv/                       # Python virtual environment
|-- app/
|   |-- backend/
|   |   |-- main.py              # FastAPI server, endpoints, and static mounts
|   |   `-- requirements.txt     # Python dependencies
|   `-- frontend/
|       `-- index.html           # Single-page dashboard & inference console
|-- Dataset/                     # Telecom customer datasets (read-only)
|   |-- churn-bigml-20.csv
|   |-- churn-bigml-80.csv
|   `-- scored_customers_test.csv
|-- Model/                       # Serialized pipeline artifacts (read-only)
|   |-- label_encoder.pkl
|   `-- telecom_churn_pipeline.pkl
|-- Notebook/                    # Exploratory data analysis & model development
|-- README.md                    # Project documentation
`-- run.bat                      # One-click launch script
```

---

## Key Features

1. **Portfolio Analytics Dashboard**
   - **Executive KPIs**: Total evaluated accounts, predicted churn rate, high-risk account volume, and average risk score.
   - **Risk Tier Segmentation**: SVG donut visualization breaking down Low Risk (<30%), Medium Risk (30-70%), and High Risk (>=70%) cohorts.
   - **Service Call Cliff Telemetry**: Custom grouped chart highlighting the sharp risk inflection occurring at 4 or more customer service inquiries.
   - **Retention Priority Watchlist**: Top at-risk accounts sorted descending by churn probability, complete with rule-based automated intervention prescriptions and one-click CSV export.

2. **Live Customer Inference Studio**
   - **Primary Churn Drivers**: Interactive controls for International Plan, Voice Mail Plan, Customer Service Calls, Total Day Minutes, State, and Area Code.
   - **Advanced Behavioral Metrics**: Collapsible panel for account length, voicemail count, evening/night/international minutes, and call frequencies.
   - **Instant Telemetry**: Probability gauge, risk tier indicators, SHAP-inspired impact factor breakdowns, and actionable retention recommendations.
   - **Cross-Component Simulation**: Directly load any customer from the Watchlist into the Inference Studio with a single click.

3. **Theme Customization**
   - Built-in Dark and Light mode toggle with persistent state saved in browser local storage.

---

## Prerequisites

- Windows 10/11
- Python 3.10 or higher
- Modern web browser (Chrome, Edge, Firefox)

---

## Quick Start

### Option 1: One-Click Run Script (Recommended)

Double-click `run.bat` in the project root, or execute it from the terminal:

```cmd
run.bat
```

The script will automatically:
1. Verify or create the `.venv` virtual environment.
2. Install all required dependencies from `app/backend/requirements.txt`.
3. Launch your default browser to `http://localhost:8000`.
4. Start the FastAPI server with live reload.

---

### Option 2: Manual Installation and Execution

1. **Create and Activate Virtual Environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```cmd
   pip install -r app/backend/requirements.txt
   ```

3. **Start the Application**:
   ```cmd
   cd app\backend
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

4. **Access the Application**:
   - Web Dashboard: `http://localhost:8000`
   - Interactive Swagger API Documentation: `http://localhost:8000/docs`
   - Alternative Redoc API Documentation: `http://localhost:8000/redoc`

---

## API Endpoints

- `GET /`: Serves the primary web dashboard.
- `GET /api/health`: Health status check and model readiness indicator.
- `GET /api/analytics`: Computes portfolio summary metrics, risk tier distribution, support call breakdown, international plan comparisons, and the priority watchlist.
- `POST /api/predict`: Runs real-time inference on submitted customer feature payloads and returns churn classification, probability score, risk tier, and an automated retention prescription.

---

## Model Evaluation Summary

- **Algorithm**: XGBoost Classifier (Trained with Scikit-learn Pipeline)
- **Accuracy**: 94.6% on independent test partition
- **F1-Score**: 0.81 for churn class detection
- **Key Predictors**: Customer service calls, international plan status, total day calling volume, and voice mail plan adoption
