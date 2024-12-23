from calendar import month
from datetime import datetime, date

import pandas as pd
from numpy.lib.utils import source

from common.config import DATA_PATH
from data_preparation.default_behaviours import MissingBehaviour
from data_preparation.sentiment_source import SentimentSource
from data_preparation.source_definition import SourceDefinition
from data_preparation.stock_source import StockSource



class DataCleaner:
    def __init__(self, sources: list[SourceDefinition], columns_to_drop: list[str] = None):
        self.sources = sources
        self.cleaner_path = f"../{DATA_PATH}/final.csv"
        self.final_date_column = 'Date'
        self.final_df = pd.DataFrame(columns=[self.final_date_column])
        self.columns_to_drop = columns_to_drop

    def custom_sort_key(self, obj):
        order = {
            MissingBehaviour.DROP.value: 0,  # 'B' comes first
            MissingBehaviour.FILL_DEFAULT.value: 1,  # 'A' comes second
            MissingBehaviour.TAKE_AVERAGE.value: 2  # 'BA' comes last
        }

        return order.get(obj.missing_behaviour.value, 3)


    def merge_sources(self, starting_date: date):
        self.sources = sorted(self.sources,  key=self.custom_sort_key, reverse=False)
        for source in self.sources:
            df = pd.read_csv(source.table_path)
            df[source.date_column] = pd.to_datetime(df[source.date_column]).dt.date
            if source.missing_behaviour.value == MissingBehaviour.DROP.value:
                df = df.dropna()
            if self.final_df.empty:
                self.final_df =  pd.merge(self.final_df, df, left_on=self.final_date_column,
                                                         right_on=source.date_column, how='outer')
                self.final_df = self.final_df[self.final_df[self.final_date_column]>starting_date]
            else:
                self.final_df = pd.merge(self.final_df, df, left_on=self.final_date_column, right_on=source.date_column, how='left')
                if source.missing_behaviour.value ==  MissingBehaviour.FILL_DEFAULT.value:
                    self.final_df[df.columns] = self.final_df[df.columns].fillna(source.default_missing)
        self.final_df = self.final_df.loc[:, ~self.final_df.columns.str.contains('^Unnamed')]
        self.final_df = self.final_df.drop(columns = self.columns_to_drop)
        self.final_df.to_csv(self.cleaner_path)


sources = [SentimentSource(), StockSource()]
columns_to_drop = ["day","link"]
data_cleaner = DataCleaner(sources = sources, columns_to_drop = columns_to_drop)
data_cleaner.merge_sources(date(year=2024,month=6,day=3))