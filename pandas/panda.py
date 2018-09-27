import pandas as pd

pd.Series
house = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv",",")[:2]

house.hist("housing_median_age")
