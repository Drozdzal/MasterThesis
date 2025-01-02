import pandas as pd

data = pd.read_csv("sentiment_gold.csv", index_col=0)
data["day"] = pd.to_datetime(data.day).dt.date
data = data.drop(columns = ["link"])
new = data.groupby("day").mean().reset_index()
new.to_csv("grouped_sentiment_gold.csv")