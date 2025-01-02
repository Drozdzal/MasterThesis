from common.config import DATA_PATH
from data_preparation.default_behaviours import MissingBehaviour
from data_preparation.source_definition import SourceDefinition


class SentimentSource(SourceDefinition):
    @property
    def table_path(self) -> str:
        return f"../{DATA_PATH}/grouped_sentiment_final_gold.csv"

    @property
    def missing_behaviour(self) -> MissingBehaviour:
        return MissingBehaviour.FILL_DEFAULT


    @property
    def default_missing(self):
        return 0

    @property
    def date_column(self):
        return "day"
