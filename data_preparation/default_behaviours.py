from enum import Enum

class MissingBehaviour(Enum):
    DROP = "DROP"
    FILL_DEFAULT = "FILL_DEFAULT"
    TAKE_AVERAGE = "TAKE_AVERAGE"
