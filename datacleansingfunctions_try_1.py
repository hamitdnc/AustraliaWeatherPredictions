# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:57:58 2021

@author: IS97853
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

WEATHERPATH = r"C:\Users\is97853\.spyder-py3\AustraliaWeatherPredictions\data"
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


def pvalue_101(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2


def drop_and_return_target(df, target_name):
    try:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    except:
        ("There is no Unnamed: 0 column")
    X = df[target_name].values
    df.drop(target_name, axis=1, inplace=True)
    yield X
    yield df


df = load_australia_weather_data(weather_path=WEATHERPATH, csv_file=CSV_FILE)
df = drop_unnamed_column(df)
print(df.columns)

y_today = df["RainToday"]
df.drop("RainToday", axis=1, inplace=True)
y_today = y_today.values

y_tomorrow = df["RainTomorrow"]
df.drop("RainTomorrow", axis=1, inplace=True)
y_tomorrow = y_tomorrow.values

df_column_location = df["Location"]
df.drop("Location", axis=1, inplace=True)

X = df.copy(deep=True).astype(int)
print(X.info())
X = X.values

est_Today = pvalue_101(X, y_today)
print(est_Today.summary())

est_tomorrow = pvalue_101(X, y_tomorrow)
print(est_Today.summary())

X_test = X[100000:]

# y_today, df = drop_and_return_target(df, TARGETNAME_RAINTODAY)
# X_today = df.values
# y_tomorrow, df = drop_and_return_target(df, TARGETNAME_RAINTOMORROW)
# X_tomorrow = df.values
