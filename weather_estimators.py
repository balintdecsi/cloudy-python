import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score


def pre_training(df):
    """Retrieve estimators for different features."""
    estimators = []
    features = ["temperature", "pressure", "humidity"]
    for i in range(len(features)):
        X = df.iloc[:,:-(3-i)]
        y = df.loc[:, features[i] + "_10"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        regr = LinearRegression()
        regr.fit(X_train, y_train)

        y_pred = regr.predict(X_test)

        estimators.append(regr)


        # print(f"{features[i]} evs: {explained_variance_score(y_test, y_pred)}, r_2: {r2_score(y_test, y_pred)}")

    return estimators

# import data_prep
# df_set, df_pred = data_prep.df_feature_extractor(data_prep.df_transformer(data_prep.json_scraper()))

# ests = pre_training(df_set)

def iterative_predictor(df_pred, estimators):
    df_pred = df_pred.values
    for idx in range(df_pred.shape[0]):
        row = df_pred[idx,:]
        y_pred = []
        for i in range(len(estimators)):
            row_pred = row[:-(3-i)]
            y_pred.append(estimators[i].predict(row_pred.reshape(1, -1)))
            row[-(3-i)] = y_pred[i]
        try:
            accumulative_y_pred = row[-((idx+1)*3):]
            df_pred[idx+1,-((idx+2)*3):-3] = accumulative_y_pred
        except IndexError:
            return accumulative_y_pred
                
# my_pred = iterative_predictor(df_pred, ests)
# print(my_pred)