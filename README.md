# Internal ML Model Tuning Framework

## Overview

This repository contains an internal machine learning model tuning framework designed to facilitate efficient hyperparameter optimization and model training. The framework consists of three main modules:

1. **DataLoader Module**: Handles loading and transforming the training and evaluation data, ensuring no data leakage.
2. **Tuner Module**: Utilizes the Optuna library for hyperparameter optimization.
3. **Model Module**: Abstracts different machine learning models, currently supporting a LightFM model wrapper.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [DataLoader Module](#dataloader-module)
  - [Tuner Module](#tuner-module)
  - [Model Module](#model-module)
- [Evaluation Methods](#evaluation-methods)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the framework, clone the repository and install the required dependencies:

```bash
pip install  git+https://github.com/ElliotNguyen68/asw-ml-tunner

```

## Usage

Here's a quick example of how to use the framework:

```python
from dataloader import DataLoader
from tuner import Tuner
from model import LightFMModel

# Load and transform data
data_loader = DataLoader(train_data_path='path/to/train_data.csv',
                         eval_data_path='path/to/eval_data.csv',
                         transform_function=your_transform_function)
train_data, eval_data = data_loader.load_data()

# Initialize the model
model = LightFMModel()

# Set up the tuner
tuner = Tuner(model=model, train_data=train_data, eval_data=eval_data)

# Run the tuning process
best_params = tuner.optimize()

# Output the best parameters
print("Best Parameters: ", best_params)
```

## Modules

### DataLoader Module

The `DataLoader` module is responsible for loading and transforming the training and evaluation data. It ensures that there is no data leakage between the training and evaluation datasets.

#### Key Features:
- Loads data from specified file paths.
- Applies the provided transformation function to the data.
- Ensures no data leakage by strictly separating training and evaluation datasets.

#### Example:

```python
from dataloader import DataLoader

def transform_function(data):
    # Define your transformation logic here
    return transformed_data

data_loader = DataLoader(train_data_path='path/to/train.csv',
                         eval_data_path='path/to/eval.csv',
                         transform_function=transform_function)
train_data, eval_data = data_loader.load_data()
```

### Tuner Module

The `Tuner` module leverages the Optuna library to perform hyperparameter optimization. It fits the provided model to the training data and evaluates it using the specified evaluation method.

#### Key Features:
- Uses Optuna for efficient hyperparameter optimization.
- Can be configured with different evaluation methods to find the best parameters.

#### Example:

```python
from aswtunner.dataloader.spark import DataLoaderSparkRecommendOOT
from aswtunner.model.lightfm import LightFMRecommenderSpark
from aswtunner.tunner.tunner_core import  Tunner
from aswtunner.metric.accurate import MapK

mapk = MapK(k=20,user_identity='contact_key',groundtruth_col='list_buy',rec_col='rec_product')

dataloader = DataLoaderSparkRecommendOOT(
    cutoff_date=pivot,
    datetime_col='transaction_date_time',
    user_identity='contact_key',
    target='brand',
    sample_ratio=.1,
    groundtruth_col='list_buy'
)

model_lightfm = LightFMRecommenderSpark(
    user_identity='contact_key',
    target='brand',
    interaction_col='quantity',
    recommend_col='rec_product',
    k=20
)

tunner = Tunner(dataloader=dataloader,model = model_lightfm)
tunner.optimize(
    values_range_dict={
        'no_components':{
                    'type':'categorical',
                    'values':[64,128,256],
                    'additional_param_optuna':{}
                },
        "learning_rate":{
            'type':'float',
                    'values':(0.011,0.5),
                    'additional_param_optuna':{
                        "step":0.01
                    }
        },
         "epochs":{
            'type':'integer',
                    'values':(1,50),
                    'additional_param_optuna':{
                    }
        },
    },
    metrics=[mapk],
    weigths=[1]
)

print("Best Parameters: ", tunner.study.best_parameters)
```

### Model Module

The `Model` module abstracts the machine learning models. Currently, it supports a LightFM model wrapper, but it can be extended to support other models.

#### Key Features:
- Provides a unified interface for different models.
- Currently supports LightFM model.

#### Example:

```python
from aswtunner.base.model import BaseModel

# The model can now be used with the Tuner module for optimization.
class BaseRecommenderModel(BaseModel)
```

## Evaluation Methods

The framework supports various evaluation methods to assess the performance of the models during the hyperparameter tuning process. These methods can be customized and extended as needed.

## Contributing

We welcome contributions to the framework. If you have any ideas, bug reports, or pull requests, please feel free to submit them.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides a high-level overview of the framework and instructions on how to get started. For more detailed documentation, please refer to the docstrings within the codebase.