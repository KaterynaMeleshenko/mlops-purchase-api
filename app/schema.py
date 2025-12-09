from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    Age: int
    Gender: int
    AnnualIncome: float
    NumberOfPurchases: int
    ProductCategory: int
    LoyaltyProgram: int          
    DiscountsAvailed: int

    TimeSpentOnWebsite: float


class PurchasePredictionResponse(BaseModel):
    will_buy: bool
    purchase_probability: float