import pandas as pd
from bs4 import BeautifulSoup
import requests


columns = ["datetime","title","source","link","top_sentiment","sentiment_score"]
df = pd.DataFrame(columns=columns)
for page in range(1,10):
    url = f"https://markets.businessinsider.com/news/nvda-stock?p={page}"

    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "lxml")
    articles = soup.find_all("div", class_="latest-news__story")
    for article in articles:
        datetime = article.find("time", class_ = "latest-news__date").get("datetime")
        title = article.find('a', class_="news-link").text
        source = article.find("span", class_ = "latest-news__source").text
        link = article.find("a",class_="news-link").get("href")
        top_sentiment = ""
        sentiment_score = 0
        df = pd.concat([pd.DataFrame([[datetime,title,source, link, top_sentiment, sentiment_score]], columns=df.columns),df], ignore_index=True)
df.to_csv("test.csv")
