from dataclasses import dataclass
from datetime import datetime


@dataclass
class BaseTickStructure:
    date: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float

    def __str__(self):
        return (f"{self.date.strftime('%y-%m-%d.%H:%M')}=O{self.open:.2f},H{self.high:.2f},"
                f"L{self.low:.2f},C{self.close:.2f}"
                f"V{self.volume:.2f}")
