# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:48:14 2021

@author: GC
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
WEATHERPATH = r"C:\Users\merve\.spyder-py3\AustraliaWeatherPredictions\data"
CSV_FILE = "CleansedDataAustraliaWeather.csv"


def load_australia_weather_data(weather_path=WEATHERPATH, csv_file=CSV_FILE):
    csv_path = os.path.join(weather_path, CSV_FILE)
    return pd.read_csv(csv_path)


def drop_unnamed_column(df):
    try:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    except:
        ("There is no Unnamed: 0 column")
    return df


def naive_bayes_model(X_train, y_train, X_test, VAR_SMOOTHING):
    gnb = GaussianNB(var_smoothing=VAR_SMOOTHING)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return y_pred, gnb


df = load_australia_weather_data(WEATHERPATH, CSV_FILE)

df = drop_unnamed_column(df)

columns = df["Location"].unique()

# print(df["Year"].unique())

df_train = pd.DataFrame(columns=columns)
df_test = pd.DataFrame(columns=columns)

# Getting data only from 2017
df_test = df[(df["Year"] < 2016)]
# print(df_test["Year"].unique())

df_train = df[(df["Year"] >= 2016)]
print(df.columns)

# - Splitting Train and Test as Numpy Array For Today
y_train_today = df_train["RainToday"].values
y_test_today = df_test["RainToday"].values

y_train_tomorrow = df_train["RainTomorrow"].values
y_test_tomorrow = df_test["RainTomorrow"].values

# drop RainToday & RainTomorrow columns
drop_columns = ["RainTomorrow", "RainToday"]
df_test.drop(drop_columns, axis=1, inplace=True)
df_train.drop(drop_columns, axis=1, inplace=True)
# Getting Location Column then drop
location_column_test = df_test["Location"]
location_column_train = df_train["Location"]
df_train.drop("Location", axis=1, inplace=True)
df_test.drop("Location", axis=1, inplace=True)

#X_train and X_test
X_train = df_train.values
X_test = df_test.values


# Using Grid Search
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
nbModel_grid = GridSearchCV(estimator=GaussianNB(),
                            param_grid=param_grid_nb,
                            verbose=1,
                            cv=10,
                            n_jobs=-1)

nbModel_grid.fit(X_train, y_train_today)
print(nbModel_grid.best_estimator_)
var_smoothing_today = nbModel_grid.best_params_["var_smoothing"]

nbModel_grid.fit(X_train, y_train_tomorrow)
print(nbModel_grid.best_estimator_)
var_smoothing_tomorrow = nbModel_grid.best_params_["var_smoothing"]

# Gaussian Naive Bayes modelling
# Best var_smoothing scores after
y_pred_today, model_today = naive_bayes_model(X_train,
                                              y_train_today,
                                              X_test,
                                              var_smoothing_today)
y_pred_tomorrow, model_tomorrow = naive_bayes_model(X_train,
                                                    y_train_tomorrow,
                                                    X_test,
                                                    var_smoothing_tomorrow)

# CV score

acc_today = accuracy_score(y_test_today, y_pred_today)
acc_tomorrow = accuracy_score(y_test_tomorrow, y_pred_tomorrow)
print(f"Acc Today  : {acc_today}\nAcc Tomorrow : {acc_tomorrow}")
