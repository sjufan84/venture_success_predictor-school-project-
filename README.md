# venture_success_predictor
A deep learning model to predict whether or not a startup will be successful or not.

## Technologies

In this project we are utilizing Python 3, Jupyter Lab, Pandas, scikit learn and hvplot.   

Pandas library -- Incredibly useful Python library for data science and data analysis  
Jupyter Lab -- Robust environment to be able to view and edit devopment projects in a streamlined system.  
hvPlot -- A high-level plotting API for the PyData ecosystem built on HoloViews.  
scikit learn -- a free software machine learning library for the Python programming language.
tensorflow -- An end-to-end open source machine learning platform

---

## Installation Guide

* Pandas -- The source code is currently hosted on GitHub at: https://github.com/pandas-dev/pandas

Binary installers for the latest released version are available at the Python Package Index (PyPI) and on Conda.

### conda
`conda install pandas`
### or PyPI
`pip install pandas`

* Jupyter Lab -- 
    [Link for detailed instructions on installing Jupyter Lab here.](https://jupyter.org/install)  
    
*  The PyViz Ecosystem (visualization package that includes hvPlot)  

### conda
`conda install -c pyviz hvplot`
### or PyPI
`pip install pyviz`  

**For more detailed information on pyviz installation and other features, please reference the [pyviz website](https://pyviz.org/)
 

*  scikit learn --  
    [Click here for link to their homepage for detailed installation instructions and other documentation](https://scikit-learn.org/stable/) 
    
---
# Imports

import pandas as pd  
from pathlib import Path  
import tensorflow as tf  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.models import Sequential  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler,OneHotEncoder  
import hvplot.pandas

## Usage

The basic usage patterns of this project are pretty straightforward.  The .csv file that contains the dataset with company information used for our projections is read in as a dataframe and then cleaned up to be used by our neural networks.  The goal is to be able to predict the success of a startup based on various factors within the dataset.  Cleanup involves eliminating unnecessary columns within our dataframe, changing categorical variables into numerical ones using OneHotEncoder, and then scaling the data using StandardScaler.  After the data is preprocessed, we split our training and testing datasets and begin to construct our models.

We used three different neural network models to compare the effectiveness of each.  The first model used two hidden layers with 'relu' activation functions and the sigmoid function for output with 100 epochs.  The second model added a third hidden layer but maintained the same activation functions.  The third model reverted back to our two hidden layer model but substiuted 'tanh' activation functions for the hidden layers.  The three models performed very similarly despite the changes, and there was not one that stood out as far superior to the others.  More detailed analysis is contained within the main notebook file.  Future models to try might include adding more epochs, further tweaking of the activation functions, or more hidden layers.  Model .h5 files can be found within the saved_models folder of this repository.

## License

Licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)  Copyright 2022 Dave Thomas.






