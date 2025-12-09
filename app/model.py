import pandas as pd
from catboost import CatBoostClassifier

from .schema import CustomerFeatures


# Загружаем модель один раз при старте приложения
model = CatBoostClassifier()
model.load_model("model/catboost_purchase_model.cbm")


def predict_purchase(features: CustomerFeatures) -> tuple[bool, float]:
    """
    Make predictions by one customer
    Return (will_buy, probability).
    """
    # Turn Pydantic-model into DataFrame
    data = pd.DataFrame([features.dict()])

    # Probability prediction
    proba = model.predict_proba(data)[0][1]  # probability of class "will_buy"

    will_buy = proba >= 0.5

    return will_buy, float(proba)