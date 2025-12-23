from dataclasses import dataclass, field
from typing import Generic, TypeVar
from enum import StrEnum

T = TypeVar("T")


@dataclass
class KafkaEnvelope(Generic[T]):
    id: str
    data: T


@dataclass
class AnalysisRequest:
    download_url: str


KafkaAnalysisRequest = KafkaEnvelope[AnalysisRequest]


class MetricType(StrEnum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


@dataclass
class Metric:
    name: str
    type: MetricType
    value: int | float | bool | str
    unit: str | None = "¤"
    description: str | None = None


@dataclass
class MetricCollection:
    provider: str
    metrics: list[Metric]
    description: str | None = None

    def __post_init__(self):
        self._keys = [m.name for m in self.metrics]

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[self._keys.index(key)]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    metrics: list[MetricCollection]


@dataclass
class BaseMsg:
    _kind: str = field(init=False)

    def __post_init__(self):
        self._kind = self.__class__.__name__


@dataclass
class ChannelMetrics(BaseMsg):
    idx: int
    segments: list[Segment]
    metrics: list[MetricCollection]

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.idx, float):
            self.idx = round(self.idx)


@dataclass
class ProgressMsg(BaseMsg):
    percent_done: int

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.percent_done, float):
            self.percent_done = round(self.percent_done)


@dataclass
class RecordingMetrics(BaseMsg):
    metrics: list[MetricCollection]


@dataclass
class ErrorMsg(BaseMsg):
    error: str
    trace: str


KafkaMsgOption = RecordingMetrics | ChannelMetrics | ProgressMsg | ErrorMsg

KafkaAnalysisResponse = KafkaEnvelope[
    KafkaMsgOption
]
