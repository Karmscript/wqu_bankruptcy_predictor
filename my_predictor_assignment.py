#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import pickle
import gzip
import json

import pandas as pd

# Create wrangle function
def wrangle(filepath):
    with gzip.open(filepath, "r") as f:
        taiwan_data = json.load(f)
    df = pd.DataFrame().from_dict(taiwan_data["observations"]).set_index("id")
    return df

#Creating make_prediction function for making predictions    
def make_predictions(data_filepath, model_filepath):
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
     #wrange data   
    X_test = wrangle(data_filepath)
    y_test = model.predict(X_test)
    y_pred_test = pd.Series(y_test, index=X_test.index, name="bankrupt")
    return y_pred_test


