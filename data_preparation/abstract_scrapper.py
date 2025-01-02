from abc import ABC, abstractmethod
from datetime import datetime
import os


class AbstractWebScrapper(ABC):
    def __init__(self, page_name):
        self._page_name = page_name
        current_time = datetime.now().strftime("%d%H%M")
        class_name = self.__class__.__name__
        self.bronze_layer_columns = ["datetime", "title", "source", "link"]
        self.silver_layer_columns = ["day","url","title","content"]

        self._run_id = f"{class_name}_{current_time}"
        os.mkdir(self._run_id)
        self._bronze_path = f"./{self._run_id}/bronze_df_2.csv"
        self._silver_path = f"./{self._run_id}/silver_df_2.csv"


    def scrap(self, start_page: int = 1, end_page: int = 10):
        self.get_article_names(start_page, end_page)
        try:
            self.get_articles_content()
        except:
            print(f"Problem with endpoint {self._page_name}")

    @abstractmethod
    def get_article_names(self,start_page: int, end_page: int):
        pass

    @abstractmethod
    def get_articles_content(self):
        pass

