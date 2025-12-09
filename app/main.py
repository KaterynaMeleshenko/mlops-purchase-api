from fastapi import FastAPI, HTTPException

from .schema import CustomerFeatures, PurchasePredictionResponse
from .model import predict_purchase

app = FastAPI(
    title="Customer Purchase Prediction API",
    description="API-service for purchase prediction",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PurchasePredictionResponse)
def predict(features: CustomerFeatures):
    try:
        will_buy, proba = predict_purchase(features)
        return PurchasePredictionResponse(
            will_buy=will_buy,
            purchase_probability=proba
        )
    except Exception as e:
        # вin real code we log it
        raise HTTPException(status_code=500, detail=str(e))