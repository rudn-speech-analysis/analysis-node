from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class KafkaEnvelope(Generic[T]):
    id: str
    data: T


@dataclass
class AnalysisRequest:
    download_url: str


KafkaAnalysisRequest = KafkaEnvelope[AnalysisRequest]


@dataclass
class AgeGenderMetrics:
    age: int
    male: float
    female: float
    child: float

    def __post_init__(self):
        if isinstance(self.age, float):
            self.age = round(self.age)


@dataclass
class EmotionMetrics:
    arousal: float
    dominance: float
    valence: float


@dataclass
class WordMetrics:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class WhisperMetrics:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[WordMetrics]


@dataclass
class SegmentMetrics:
    emotion: EmotionMetrics
    whisper: WhisperMetrics


@dataclass
class ChannelMetrics:
    idx: int
    age_gender: AgeGenderMetrics
    talk_percent: float
    segments: list[SegmentMetrics]

    def __post_init__(self):
        if isinstance(self.idx, float):
            self.idx = round(self.idx)


@dataclass
class ProgressMsg:
    percent_done: int

    def __post_init__(self):
        if isinstance(self.percent_done, float):
            self.percent_done = round(self.percent_done)


@dataclass
class AudioMetrics:
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int
    max_dbfs: float
    rms: int
    raw_data_length: int


@dataclass
class RecordingMetrics:
    audio: AudioMetrics


@dataclass
class ErrorMsg:
    error: str
    trace: str


KafkaAnalysisResponse = KafkaEnvelope[
    RecordingMetrics | ChannelMetrics | ProgressMsg | ErrorMsg
]
