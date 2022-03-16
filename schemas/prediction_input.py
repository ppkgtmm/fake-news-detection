from typing import List
from pydantic import BaseModel


class PredictionInput(BaseModel):
    texts: List[str]
