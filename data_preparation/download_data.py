import yfinance as yf
from datetime import datetime

from data_preparation.examle_tickers import MOCKED_TICKERS


class DataDownloader:
    def __init__(self, tickers: dict):
        self.tickers = tickers

    def donwload_data(self, start_date: datetime, end_date: datetime):
        date_format = "%Y-%m-%d"
        start = start_date.strftime(date_format)
        end = end_date.strftime(date_format)
        data = yf.download(list(self.tickers.values()), start=start, end=end)
        data.to_csv("test.csv")

data_downloader = DataDownloader(MOCKED_TICKERS)
start_date = datetime(year = 2020, month= 1,day = 1)
end_date = datetime(year = 2024, month= 1,day = 1)


data_downloader.donwload_data(start_date, end_date)

