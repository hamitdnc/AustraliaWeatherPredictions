# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:02:13 2021

@author: IS97853
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

WEATHERPATH = r"C:\Users\merve\.spyder-py3\AustraliaWeatherPredictions\data"
CSV_FILE = "weatherAUS.csv"
TARGETNAME_RAINTODAY = "RainToday"
TARGETNAME_RAINTOMORROW = "RainTomorrow"

# Loading dataset from file


def load_australia_weather_data(weather_path=WEATHERPATH, csv_file=CSV_FILE):
    csv_path = os.path.join(weather_path, csv_file)
    return pd.read_csv(csv_path)


# def train_test_splitter(df, TEST_SIZE=0.2, RANDOM_STATE=42):
#     train_set, test_set = train_test_split(
#         df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
#     return train_set, test_set


def drop_columns_sum_over_fiftythousands(df):
    columns = df.columns
    print(f"Columns : {columns}")
    columns_to_drop = []
    for column in columns:
        if df[column].isnull().sum() > 50000:
            columns_to_drop.append(column)
    print(columns_to_drop)
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df


def drop_all_na_rows(df):
    df.dropna(inplace=True)
    return df


def drop_and_return_target(df, target_name, nontarget_name):
    try:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    except:
        ("There is no Unnamed: 0 column")
    X = df[target_name].values
    df.drop([target_name, nontarget_name], axis=1, inplace=True)
    yield X
    yield df


def pvalue_101(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2


def columns_to_get_dummies(df, columns_to_encode):
    df = pd.get_dummies(df, columns=columns_to_encode,
                        prefix_sep="__", drop_first=True)
    return df


def label_encoding(df, columns_to_label_encode):
    for label in columns_to_label_encode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label])

    # def acc_score(y_test, y_pred):
    #     ac = accuracy_score(y_test, y_pred)
    #     print(ac)


def deal_with_date(date_value):
    pass


df = load_australia_weather_data(WEATHERPATH, CSV_FILE)

df = drop_columns_sum_over_fiftythousands(df)
df['Date'] = df['Date'].astype('datetime64[ns]').astype(str)
df[["Year", "Month", "Day"]] = df['Date'].str.split("-",expand=True)
df.drop("Date", axis=1, inplace=True)
print(df["Year"].unique())
drop_all_na_rows(df)
# print(df.info())

columns_to_encode = ["WindGustDir", "WindDir9am", "WindDir3pm"]
df = columns_to_get_dummies(df, columns_to_encode)
print(df.shape)

columns_to_label_encode = ["RainTomorrow", "RainToday"]
label_encoding(df, columns_to_label_encode)
print(df.shape)
df.to_csv("CleansedDataAustraliaWeather.csv")


# # print(df.info())
# # print(df.shape)
# # my_df_correlation = df.corr(method='pearson', min_periods=19)
# # df.to_csv("cleansed_data.csv")
# # train_set, test_set = train_test_splitter(df, 0.2, 42)
