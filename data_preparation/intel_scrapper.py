import pandas as pd
import requests
from bs4 import BeautifulSoup

from data_preparation.abstract_scrapper import AbstractWebScrapper

class IntelWebScrapper(AbstractWebScrapper):
    def get_articles_content(self):
        bronze_df = pd.read_csv(self._bronze_path) #["datetime", "title", "source", "link"]
        silver_df = pd.DataFrame(columns=self.silver_layer_columns) #["day","url","title","content"]
        for index, row in bronze_df.iterrows():
            try:
                source = row["source"]
                if source in ["RTTNews", "TipRanks"]:
                    url = "https://markets.businessinsider.com" + row["link"]
                    try:
                        full_text = self.get_article_content(url, source)
                        silver_df = pd.concat([pd.DataFrame([[row["datetime"], row["link"], row["title"], full_text]],
                                                 columns=self.silver_layer_columns), silver_df], ignore_index=True)


                        print(f"dumped {url}")
                    except:
                        print(f"Couldnt dump for for {url}")
                else:
                    url = row["link"]
            except:
                pass
        silver_df.to_csv(self._silver_path)

    def get_article_content(self, url: str, source: str):
        session = requests.Session()

        # Set headers for the session
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        })

        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        full_text = ""
        try:
            if source == "Seeking Alpha":
                #CANNOT PARSE SEEKING ALPHA
                pass
            elif source in ["TipRanks","RTTNews"]:
                articles = soup.find("div", class_="col-md-8")
                raw_paragraphs = articles.find_all("p")
                full_text = ""
                for raw_paragraph in raw_paragraphs:
                    full_text += " " + raw_paragraph.text
            return full_text
        except:
            return None

    def get_article_names(self,start_page: int, end_page: int):
        bronze_df = pd.DataFrame(columns=self.bronze_layer_columns)
        for page in range(start_page, end_page):
            url = self._page_name + str(page)
            response = requests.get(url)
            html = response.text
            soup = BeautifulSoup(html, "lxml")
            articles = soup.find_all("div", class_="latest-news__story")
            for article in articles:
                datetime = article.find("time", class_="latest-news__date").get("datetime")
                title = article.find('a', class_="news-link").text
                source = article.find("span", class_="latest-news__source").text
                link = article.find("a", class_="news-link").get("href")
                bronze_df = pd.concat([pd.DataFrame([[datetime, title, source, link]],
                                             columns=bronze_df.columns), bronze_df], ignore_index=True)
        bronze_df.to_csv(self._bronze_path)


intel_scrapper = IntelWebScrapper("https://markets.businessinsider.com/news/intc-stock?p=")
intel_scrapper.get_article_names(1,40)
intel_scrapper.get_articles_content()
