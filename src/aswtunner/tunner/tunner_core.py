from abc import ABC
from typing import List, Any, Dict
from functools import partial

from aswtunner.base.metric import BaseMetric
from aswtunner.base.dataloader import BaseDataLoader
from aswtunner.base.model import BaseModel

import optuna

class Tunner(ABC):
    study:optuna.Study
    def __init__(self,model:BaseModel,dataloader:BaseDataLoader) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
    
    def _get_parameter_trial_optuna(
        self,
        trial,
        values_range_dict:Dict 
    ):
        dict_param = {}
        for key,value in values_range_dict.items():
            # print(key,value)
            if value['type']=='float':
                param_value = trial.suggest_float(key,value['values'][0],value['values'][1],**value['additional_param_optuna'])
            elif value['type'] == 'integer':
                param_value = trial.suggest_int(key,value['values'][0],value['values'][1],**value['additional_param_optuna'])
            elif value['type'] == 'categorical':
                param_value = trial.suggest_categorical(key,value['values'],**value['additional_param_optuna'])
            dict_param[key]=param_value
        return dict_param
                
        
    def optimize(
        self,
        values_range_dict:Dict,
        metrics: List[BaseMetric],
        weigths : List[float] = None,
        n_trials: int =20
    ):
        """

        Args:
            values_range_dict (Dict): {
                'att1':{
                    'type':'float',
                    'values':(0.1,0.2),
                    'additional_param_optuna':{}
                },
                'att4':{
                    'type':'integer',
                    'values':(1,20),
                    'additional_param_optuna':{}
                },
                'att2':{
                    'type':'categorical',
                    'values':(True, False, None),
                    'additional_param_optuna':{}
                }
            }
        """
        if weigths is not None:
            assert len(weigths)== len(metrics)
        else:
            weigths = [1 for _ in range(len(metrics))]
        
        step_wrapper = partial(self.step, metrics,values_range_dict,weigths)
        
        study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())
        self.study = study
        study.optimize(step_wrapper, n_trials=n_trials)
    
    def step(
        self,
        metrics: List[BaseMetric],
        values_range_dict: Dict,
        weigths : List[float],
        trial 
    ):
        
        model_parameters = self._get_parameter_trial_optuna(trial=trial,values_range_dict=values_range_dict)
        
        self.model(**model_parameters)
        data_train = self.dataloader.get_train_data()
        data_eval = self.dataloader.get_validate_data()
        
        self.model.fit(
            data_train
        )
        
        result = self.model.predict(data_eval)
        
        metric_sofar =0
        for w,metric in zip(weigths,metrics):
            res = metric.evaluate(result,data_eval)
            metric_sofar+= w*res
        
        
        return metric_sofar
        