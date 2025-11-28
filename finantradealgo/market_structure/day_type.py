from enum import IntEnum

class DAY_TYPE(IntEnum):
    TREND = 1
    NORMAL = 0
    LIQUIDATION = -1
    # RANGE vs NORMAL will be handled by logic