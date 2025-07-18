from pydantic import BaseModel, Field
from typing import List, Optional

class MatchInput(BaseModel):
    features: List[float] = Field(..., example=[0.23, 0.56, 0.78, 0.12])

class MatchOutput(BaseModel):
    prediction: int = Field(..., example=1)
    probability: float = Field(..., example=0.89)
