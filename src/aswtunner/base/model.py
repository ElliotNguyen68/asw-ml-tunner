from abc import ABC
from abc import abstractmethod

import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def predict():
        raise NotImplemented
    
    @abstractmethod
    def fit():
        raise NotImplemented


class BaseRecommenderModel(BaseModel):
    user_identity:str
    target: str
    def __init__(self, user_identity:str, target:str) -> None:
        super().__init__()
        self.user_identity = user_identity
        self.target = target 
    
    
    
   