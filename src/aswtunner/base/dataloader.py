from abc import ABC
from abc import abstractmethod
from typing import Callable


class BaseDataLoader(ABC):
    transformation_function: Callable 
    is_fitted: bool = False
    
    @abstractmethod
    def fit():
        raise NotImplemented
    
    def transform(self,*args, **kwargs):
        return self.transformation_function(*args,**kwargs)
    
    def set_transform(self,transform: Callable):
        self.transformation_function = transform

    @abstractmethod
    def get_train_data(self,):
        assert self.is_fitted

    @abstractmethod
    def get_validate_data(self):
        assert self.is_fitted
        
    def sample(self):
        raise NotImplemented
