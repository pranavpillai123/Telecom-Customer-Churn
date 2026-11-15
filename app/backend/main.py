import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Base directories relative to app/backend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "Model"))
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "Dataset"))
DATASET_DIR_LOWER = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "dataset"))

PIPELINE_PATH = os.path.join(MODEL_DIR, "telecom_churn_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCORED_DATA_PATH = os.path.join(DATASET_DIR, "scored_customers_test.csv")
SCORED_DATA_FALLBACK_1 = os.path.join(DATASET_DIR_LOWER, "scored_customers_test.csv")
RAW_DATA_FALLBACK = os.path.join(DATASET_DIR, "churn-bigml-20.csv")
RAW_DATA_FALLBACK_2 = os.path.join(DATASET_DIR_LOWER, "churn-bigml-20.csv")

# Global memory storage for model and scored dataset
pipeline = None
label_encoder = None
scored_df: pd.DataFrame = None


def get_action_suggestion(intl_plan: int, cs_calls: int) -> str:
    """Generate retention action based on business rules."""
    if intl_plan == 1:
        return "Audit International Plan & Offer Roaming Discount"
    elif cs_calls >= 4:
        return "Escalate to Senior Support & Waive Next Fee"
    else:
        return "Proactive Retention Outreach Call"


def assign_risk_tier(score: float) -> str:
    if score >= 70.0:
        return "High Risk"
    elif score >= 30.0:
        return "Medium Risk"
    else:
        return "Low Risk"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, label_encoder, scored_df
    # 1. Load pipeline and label encoder
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # 2. Load or compute scored dataset
    if os.path.exists(SCORED_DATA_PATH):
        scored_df = pd.read_csv(SCORED_DATA_PATH)
    elif os.path.exists(SCORED_DATA_FALLBACK_1):
        scored_df = pd.read_csv(SCORED_DATA_FALLBACK_1)
    else:
        raw_path = RAW_DATA_FALLBACK if os.path.exists(RAW_DATA_FALLBACK) else RAW_DATA_FALLBACK_2
        if not os.path.exists(raw_path):
            raise FileNotFoundError("Neither scored dataset nor fallback churn-bigml-20.csv found.")
        
        raw_df = pd.read_csv(raw_path)
        # Drop charge columns if present
        cols_to_drop = [c for c in ["Total day charge", "Total eve charge", "Total night charge", "Total intl charge"] if c in raw_df.columns]
        cleaned_df = raw_df.drop(columns=cols_to_drop)

        # Standardize boolean/string fields
        if "International plan" in cleaned_df.columns and cleaned_df["International plan"].dtype == object:
            cleaned_df["International plan"] = cleaned_df["International plan"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
        if "Voice mail plan" in cleaned_df.columns and cleaned_df["Voice mail plan"].dtype == object:
            cleaned_df["Voice mail plan"] = cleaned_df["Voice mail plan"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
        if "Area code" in cleaned_df.columns:
            cleaned_df["Area code"] = cleaned_df["Area code"].astype(str)

        # Drop target if present for prediction
        target_col = "Churn" if "Churn" in cleaned_df.columns else None
        features = cleaned_df.drop(columns=[target_col]) if target_col else cleaned_df

        probas = pipeline.predict_proba(features)[:, 1]
        cleaned_df["risk_score_%"] = np.round(probas * 100, 2)
        cleaned_df["risk_tier"] = cleaned_df["risk_score_%"].apply(assign_risk_tier)
        scored_df = cleaned_df

    yield


app = FastAPI(
    title="Telecom Customer Churn Prediction & Analytics API",
    description="Production API serving telecom churn risk scoring, analytics, and retention intelligence.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    State: str = Field(default="NY", description="US State code (e.g. 'NY', 'CA')")
    Account_length: int = Field(default=100, alias="Account length", description="Number of days the account has been active")
    Area_code: Union[str, int] = Field(default="415", alias="Area code", description="Area code (408, 415, 510)")
    International_plan: Union[str, int] = Field(default="No", alias="International plan", description="'Yes'/'No' or 1/0")
    Voice_mail_plan: Union[str, int] = Field(default="No", alias="Voice mail plan", description="'Yes'/'No' or 1/0")
    Number_vmail_messages: int = Field(default=0, alias="Number vmail messages", description="Number of voicemail messages")
    Total_day_minutes: float = Field(default=180.0, alias="Total day minutes", description="Total daytime call minutes")
    Total_day_calls: int = Field(default=100, alias="Total day calls", description="Total daytime calls count")
    Total_eve_minutes: float = Field(default=200.0, alias="Total eve minutes", description="Total evening call minutes")
    Total_eve_calls: int = Field(default=100, alias="Total eve calls", description="Total evening calls count")
    Total_night_minutes: float = Field(default=200.0, alias="Total night minutes", description="Total nighttime call minutes")
    Total_night_calls: int = Field(default=100, alias="Total night calls", description="Total nighttime calls count")
    Total_intl_minutes: float = Field(default=10.0, alias="Total intl minutes", description="Total international call minutes")
    Total_intl_calls: int = Field(default=4, alias="Total intl calls", description="Total international calls count")
    Customer_service_calls: int = Field(default=1, alias="Customer service calls", description="Number of support inquiries")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "State": "NY",
                "Account length": 100,
                "Area code": "415",
                "International plan": "No",
                "Voice mail plan": "No",
                "Number vmail messages": 0,
                "Total day minutes": 180.0,
                "Total day calls": 100,
                "Total eve minutes": 200.0,
                "Total eve calls": 100,
                "Total night minutes": 200.0,
                "Total night calls": 100,
                "Total intl minutes": 10.0,
                "Total intl calls": 4,
                "Customer service calls": 1,
            }
        },
    }


@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "healthy",
        "service": "Telecom Churn Prediction & Analytics API",
        "model_loaded": pipeline is not None,
        "dataset_loaded": scored_df is not None,
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Telecom Churn Prediction & Analytics API",
        "model_loaded": pipeline is not None,
        "dataset_loaded": scored_df is not None,
    }


@app.get("/api/analytics")
def get_analytics():
    if scored_df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    total_customers = len(scored_df)
    if total_customers == 0:
        raise HTTPException(status_code=400, detail="Empty dataset.")

    # KPIs
    if "risk_score_%" in scored_df.columns:
        churn_count = int((scored_df["risk_score_%"] >= 50.0).sum())
        churn_rate = round(float(churn_count / total_customers * 100), 2)
        avg_risk_score = round(float(scored_df["risk_score_%"].mean()), 2)
    elif "Churn" in scored_df.columns:
        churn_count = int((scored_df["Churn"] == True).sum())
        churn_rate = round(float(churn_count / total_customers * 100), 2)
        avg_risk_score = 0.0
    else:
        churn_rate = 0.0
        avg_risk_score = 0.0

    high_risk_count = int((scored_df["risk_tier"] == "High Risk").sum()) if "risk_tier" in scored_df.columns else 0

    kpis = {
        "total_customers": total_customers,
        "predicted_churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "avg_risk_score": avg_risk_score,
    }

    # Risk distribution
    low_risk = int((scored_df["risk_tier"] == "Low Risk").sum()) if "risk_tier" in scored_df.columns else 0
    med_risk = int((scored_df["risk_tier"] == "Medium Risk").sum()) if "risk_tier" in scored_df.columns else 0
    high_risk = high_risk_count

    risk_distribution = {
        "Low Risk": low_risk,
        "Medium Risk": med_risk,
        "High Risk": high_risk,
    }

    # Churn by Customer service calls (0 to 9)
    churn_by_service_calls = []
    cs_grouped = scored_df.groupby("Customer service calls")["risk_score_%"].agg(["count", "mean"]).reset_index()
    for _, row in cs_grouped.iterrows():
        churn_by_service_calls.append({
            "service_calls": int(row["Customer service calls"]),
            "count": int(row["count"]),
            "avg_risk_score": round(float(row["mean"]), 2),
        })

    # Churn by International plan (1 vs 0)
    intl_grouped = scored_df.groupby("International plan")
    intl_summary = {}
    for plan_val, group in intl_grouped:
        plan_key = "Yes" if plan_val in [1, "Yes", "yes"] else "No"
        group_total = len(group)
        group_churn = int((group["risk_score_%"] >= 50.0).sum()) if "risk_score_%" in group.columns else 0
        intl_summary[plan_key] = {
            "count": group_total,
            "avg_risk_score": round(float(group["risk_score_%"].mean()), 2) if "risk_score_%" in group.columns else 0.0,
            "churn_rate": round(float(group_churn / group_total * 100), 2) if group_total > 0 else 0.0,
        }

    # High-risk watchlist: top 15 customers sorted descending by risk_score_%
    top15 = scored_df.sort_values(by="risk_score_%", ascending=False).head(15)
    watchlist = []
    for _, cust in top15.iterrows():
        intl_val = int(cust.get("International plan", 0))
        cs_val = int(cust.get("Customer service calls", 0))
        action = get_action_suggestion(intl_val, cs_val)
        watchlist.append({
            "State": str(cust.get("State", "")),
            "Area code": str(cust.get("Area code", "")),
            "Account length": int(cust.get("Account length", 0)),
            "International plan": "Yes" if intl_val == 1 else "No",
            "Customer service calls": cs_val,
            "Total day minutes": round(float(cust.get("Total day minutes", 0.0)), 1),
            "risk_score_%": round(float(cust.get("risk_score_%", 0.0)), 2),
            "risk_tier": str(cust.get("risk_tier", "High Risk")),
            "action_suggestion": action,
        })

    return {
        "kpis": kpis,
        "risk_distribution": risk_distribution,
        "churn_by_service_calls": churn_by_service_calls,
        "churn_by_intl_plan": intl_summary,
        "high_risk_watchlist": watchlist,
    }


@app.post("/api/predict")
def predict_churn(payload: CustomerFeatures):
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model pipeline not initialized.")

    data = payload.model_dump(by_alias=True)

    # 1. Cast Area code to str
    data["Area code"] = str(data["Area code"])

    # 2. Map International plan to 1 or 0
    if isinstance(data["International plan"], str):
        data["International plan"] = 1 if data["International plan"].strip().lower() in ["yes", "y", "1", "true"] else 0
    else:
        data["International plan"] = 1 if int(data["International plan"]) == 1 else 0

    # 3. Map Voice mail plan to 1 or 0
    if isinstance(data["Voice mail plan"], str):
        data["Voice mail plan"] = 1 if data["Voice mail plan"].strip().lower() in ["yes", "y", "1", "true"] else 0
    else:
        data["Voice mail plan"] = 1 if int(data["Voice mail plan"]) == 1 else 0

    # Ensure exact 15-column schema required by pipeline
    ordered_columns = [
        "State",
        "Account length",
        "Area code",
        "International plan",
        "Voice mail plan",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total eve minutes",
        "Total eve calls",
        "Total night minutes",
        "Total night calls",
        "Total intl minutes",
        "Total intl calls",
        "Customer service calls",
    ]

    input_df = pd.DataFrame([{col: data[col] for col in ordered_columns}])

    try:
        prob = float(pipeline.predict_proba(input_df)[:, 1][0])
        pred_class = pipeline.predict(input_df)[0]
        label = label_encoder.inverse_transform([pred_class])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    risk_score_percent = round(prob * 100, 2)
    risk_tier = assign_risk_tier(risk_score_percent)
    churn_pred = "Churn" if bool(label) is True or risk_score_percent >= 50.0 else "Stay"

    # Dynamic recommendation tailored to input attributes
    if risk_tier == "High Risk":
        if data["Customer service calls"] >= 4:
            recommendation = (
                f"Urgent: Customer has {data['Customer service calls']} support inquiries. "
                "Immediately route to dedicated retention specialist and offer service credits."
            )
        elif data["International plan"] == 1:
            recommendation = (
                "High churn probability on international plan. Proactively review roaming rates and offer an upgraded global bundle."
            )
        elif data["Total day minutes"] > 250:
            recommendation = (
                f"Heavy daytime user ({data['Total day minutes']} mins). Offer an unlimited high-tier corporate plan to prevent attrition."
            )
        else:
            recommendation = (
                "High churn risk detected. Schedule immediate proactive customer success check-in with promotional loyalty incentives."
            )
    elif risk_tier == "Medium Risk":
        if data["Customer service calls"] >= 3:
            recommendation = "Moderate churn risk with elevated support tickets. Follow up within 48 hours to confirm inquiry satisfaction."
        elif data["Total day minutes"] > 220:
            recommendation = "Moderate usage churn risk. Suggest plan optimization tailored to high daytime calling volume."
        else:
            recommendation = "Moderate churn vulnerability. Enroll customer in quarterly loyalty reward engagement sequence."
    else:
        recommendation = "Low churn risk. Customer relationship is healthy and stable; maintain standard engagement."

    return {
        "churn_prediction": churn_pred,
        "risk_score_percent": risk_score_percent,
        "risk_tier": risk_tier,
        "recommendation": recommendation,
    }


# Mount frontend static directory if present
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
