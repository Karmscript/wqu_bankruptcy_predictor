#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries here
import gzip
import json
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


# In[ ]:


# Load data file
with gzip.open("data/taiwan-bankruptcy-data.json.gz", "r") as f:
    taiwan_data = json.load(f)
print(type(taiwan_data))

len(taiwan_data["observations"])
print(taiwan_data['metadata'])
#Getting the keys in the datasets
taiwan_data_keys = taiwan_data.keys()
print(taiwan_data_keys)

#Getting the number of observations in the datasets
n_companies = len(taiwan_data["observations"])
print(n_companies)

#getting the number of data in one observation
n_features = len(taiwan_data["observations"][0])
print(n_features)



# In[ ]:


#Wrangle function for wrangling files
# Create wrangle function
def wrangle(filepath):
    with gzip.open(filepath, "r") as f:
        taiwan_data = json.load(f)
    df = pd.DataFrame().from_dict(taiwan_data["observations"]).set_index("id")
    return df


# In[ ]:


#Exploring the data
nans_by_col = pd.Series(df.isnull().sum())  
print("nans_by_col shape:", nans_by_col.shape)
nans_by_col.head()

# Plot class balance
fig, ax=plt.subplots()
df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    ax=ax,
    xlabel= "Bankrupt",
    ylabel= "Frequency",
    title = "Class Balance"
);


# In[ ]:


#Splitting the data
target = "bankrupt"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[ ]:


#Resampling using Over sampler
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()


# In[ ]:


#Building model/classifier
clf = RandomForestClassifier(random_state=42)
#Cross validation scores
cv_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
print(cv_scores)
#Grid search
model = GridSearchCV(
 clf,
  param_grid = params,
    cv = 5,
    n_jobs = -1,
    verbose=1
)

#Fitting model
model.fit(X_train_over, y_train_over)

cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(5)
best_params = model.best_params_
print(best_params)


# In[ ]:


#Evaluating the model
acc_train = model.score(X_train, y_train)
acc_test = model.score(X_test, y_test)

print("Model Training Accuracy:", round(acc_train, 4))
print("Model Test Accuracy:", round(acc_test, 4))

# Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax);

#classifiction report
class_report = classification_report(y_test, model.predict(X_test))
print(class_report)


# In[ ]:


#Communicating results
 pd.Series(model.best_estimator_.feature_importances_, X_train_over.columns).sort_values()

# plotting the feature importances
importances = model.best_estimator_.feature_importances_
fig, ax=plt.subplots()
feat_imp =pd.Series(importances, index=X_train_over.columns).sort_values(ascending=True)
(feat_imp.tail(10)).plot(kind="barh", ax=ax)
ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance");


# In[ ]:


#saving model
# Save model
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)

#using the model
# Import your module
import pickle
import gzip
import json

import pandas as pd
from my_predictor_assignment import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()

