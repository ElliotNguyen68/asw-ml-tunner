from abc import ABC
from abc import abstractmethod
from typing import List

import pandas as pd


class BaseModel(ABC):
    model_parameters: List[str] = None
    fit_parameters: List[str] = None

    @abstractmethod
    def predict():
        raise NotImplemented

    def fit(self):
        assert self.model_parameters is not None and self.fit_parameters is not None


class BaseRecommenderModel(BaseModel):
    user_identity: str
    target: str

    def __init__(self, user_identity: str, target: str) -> None:
        super().__init__()
        self.user_identity = user_identity
        self.target = target
