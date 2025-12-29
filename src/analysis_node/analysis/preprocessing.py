import logging
from functools import partial
from typing import Generator, Tuple
import pydub
import tempfile
import pathlib
import dacite
from analysis_node.config import Config
from analysis_node.messages import Metric, MetricType, MetricCollection
import copy

from dataclasses import dataclass


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


logger = logging.getLogger(__name__)


def split_audio(input_file) -> list[tempfile._TemporaryFileWrapper]:
    audio = pydub.AudioSegment.from_file(input_file)

    try:
        split = audio.split_to_mono()
    except ValueError as ex:
        raise ValueError("Input audio file must be stereo.") from ex

    def export(channel) -> tempfile._TemporaryFileWrapper:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav")
        channel.export(tmp_file, format="wav")
        return tmp_file

    channels = list(map(export, split))
    return channels


def filger_segments(segment_data: WhisperMetrics, config: Config) -> bool:
    cfg = config.values["preprocessing"]
    # Removing short segments
    if (segment_data.end - segment_data.start) < cfg["min_segment_length_sec"]:
        return False

    # Removing segments that contain stop_phrases
    matching_stop_phrases = [
        s for s in cfg["stop_phrases"] if s.lower() in segment_data.text.lower()
    ]
    if matching_stop_phrases:
        # Remove the segment if it contains only a stop phrase
        if max(map(len, matching_stop_phrases)) + cfg["stop_phrase_length_delta"] > len(
            segment_data.text.strip()
        ):
            return False
    if segment_data.no_speech_prob > cfg["no_speech_threshold"]:
        return False

    return True


def aggregate_group(merged_seg: WhisperMetrics, group: list[WhisperMetrics]):
    """
    Aggregates a group of segments into the merged_seg dict.
    """

    # Helper to check if a key is averagable (float-type metrics)
    AVERAGABLE_KEYS = {
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
        "temperature",
    }

    # Helper to check concatenatable lists
    CONCAT_KEYS = {"words", "tokens"}

    if len(group) == 1:
        return  # Nothing to aggregate

    merged_seg.id = min(seg.id for seg in group)

    merged_seg.start = min(seg.start for seg in group)
    merged_seg.end = max(seg.end for seg in group)
    merged_seg.text = " ".join(seg.text for seg in group)
    merged_seg.seek = min(seg.seek for seg in group)

    for key in CONCAT_KEYS:
        concat_val = []
        for seg in group:
            concat_val.extend(getattr(seg, key))
        setattr(merged_seg, key, concat_val)

    # Average floats
    for key in AVERAGABLE_KEYS:
        values = [getattr(seg, key, 0.0) for seg in group]
        if values:
            avg = sum(values) / len(values)
            setattr(merged_seg, key, avg)


def merge_close_segments(
    segments: list[WhisperMetrics], min_segment_distance_sec: float
) -> list[WhisperMetrics]:
    if not segments:
        return []

    merged = [copy.deepcopy(segments[0])]
    current_group = [segments[0]]

    for current in segments[1:]:
        previous = merged[-1]

        gap = current.start - previous.end
        if gap < min_segment_distance_sec:
            current_group.append(current)
        else:
            aggregate_group(previous, current_group)

            merged.append(copy.deepcopy(current))
            current_group = [current]

    aggregate_group(merged[-1], current_group)
    return merged


def segmentize(
    source: pathlib.Path | str,
    whispermodel,
    config: Config,
) -> Generator[Tuple[WhisperMetrics, pathlib.Path]]:
    transcription = whispermodel.transcribe(
        str(source),
        word_timestamps=True,
        language="ru",
    )

    segments = list(
        filter(
            partial(filger_segments, config=config),
            [
                dacite.from_dict(
                    data_class=WhisperMetrics,
                    data=s,
                )
                for s in transcription["segments"]
            ],
        )
    )

    min_segment_distance_sec = config.values["preprocessing"][
        "min_segment_distance_sec"
    ]
    if isinstance(min_segment_distance_sec, bool) and not min_segment_distance_sec:
        merged_segments = segments
    else:
        merged_segments = merge_close_segments(segments, min_segment_distance_sec)

    audio = pydub.AudioSegment.from_file(source)
    logger.info(f"Segmentizing {source} with {len(transcription["segments"])} segments")
    for segment_data in merged_segments:

        start = float(segment_data.start) * 1000
        end = float(segment_data.end) * 1000
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            logger.debug(f"Writing segment [{start}, {end}] to {tmp_file.name}")
            audio[start:end].export(tmp_file.name, format="wav")
            yield (segment_data, pathlib.Path(tmp_file.name))


def get_audio_metrics(audio_file: pathlib.Path | str) -> MetricCollection:
    audio = pydub.AudioSegment.from_file(
        audio_file
    )  # Replace with your file path and extension
    duration_seconds = (
        len(audio) / 1000.0
    )  # Duration in seconds (pydub uses milliseconds internally)
    sample_rate = audio.frame_rate  # Sample rate (Hz)
    channels = audio.channels  # Number of channels (1 for mono, 2 for stereo)
    bit_depth = audio.sample_width * 8  # Bit depth (e.g., 16-bit, 24-bit)
    max_dbfs = (
        audio.max_dBFS
    )  # Maximum loudness in dBFS (decibels relative to full scale)
    rms = audio.rms  # Root mean square (average power/amplitude)
    raw_data_length = len(audio.raw_data)  # Length of raw audio bytes

    return MetricCollection(
        "pydub",
        [
            Metric("duration", MetricType.FLOAT, duration_seconds, "seconds"),
            Metric("sample rate", MetricType.INT, sample_rate, "Hz"),
            Metric(
                "channels",
                MetricType.INT,
                channels,
                None,
            ),
            Metric("bit depth", MetricType.INT, bit_depth, "bits/sample"),
            Metric("max loudness", MetricType.FLOAT, max_dbfs, "dBFS"),
            Metric(
                "rms",
                MetricType.INT,
                rms,
                None,
            ),
            Metric("raw data length", MetricType.INT, raw_data_length, "samples"),
        ],
    )
