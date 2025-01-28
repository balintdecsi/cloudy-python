import os.path
from urllib.request import urlopen
import pandas as pd
import json
import numpy as np

def synthetic_importer():
    path = os.path.join("..", "weather-logger-django", "syntheticdata.csv")
    df = pd.read_csv(path)

    return df

def json_scraper():
    """Scrapes collected weather data from proejct's server through REST API."""
    json_dir = {}
    with urlopen("http://itk.rocks/weather/?format=json") as response:
        for line in response:
            line_dir = json.loads(line)
            for entry in line_dir:
                for k, v in entry.items():
                    try:
                        json_dir[k].append(v)
                    except KeyError:
                        json_dir[k] = [v]
            
    json_df = pd.DataFrame(json_dir)


    return json_df

def df_transformer(df):
    """Transforms the collected DataFrame so it splits datetime expression into month, day and hour."""
    created = df.created.str.split("-", expand=True)
    month = created.iloc[:,[1]].applymap(lambda x: int(x))
    month.columns = ["month"]
    created_2 = created.iloc[:,-1].str.split("T", expand=True)
    day = created_2.iloc[:,[0]].applymap(lambda x: int(x))
    day.columns = ["day"]
    created_3 = created_2.iloc[:,-1].str.split(":", expand=True)
    hour = created_3.iloc[:,[0]].applymap(lambda x: int(x))
    hour.columns = ["hour"] 

    return pd.concat([month, day, hour, df.drop("created", axis=1)], axis=1)

def df_feature_extractor(df):
    """"Creates columns for each day: temperature, pressure and humidity each of the following 10 days."""
    for i in range(1, 11):
        for feature in ["temperature", "pressure", "humidity"]:
            feature_col = np.concatenate((df[feature][i:], np.array(i * [np.nan])), axis=0)
            df.loc[:, feature + "_" + str(i)] = feature_col
    
    return df[:-10], df[-10:]