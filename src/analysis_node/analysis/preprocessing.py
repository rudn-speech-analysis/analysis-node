from typing import Generator, Tuple
import pydub
import tempfile
import pathlib
from analysis_node.config import Config
from analysis_node.messages import Metric, MetricType, MetricCollection


def split_audio(input_file) -> Tuple[pathlib.Path, pathlib.Path]:
    audio = pydub.AudioSegment.from_file(input_file)

    try:
        split = audio.split_to_mono()
    except ValueError as ex:
        raise ValueError("Input audio file must be stereo.") from ex

    def export(channel) -> pathlib.Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            channel.export(tmp_file, format="wav")
        return pathlib.Path(tmp_file.name)

    left, right = list(map(export, split))
    return left, right


def filter_out(segment_data: dict, config: Config) -> bool:
    cfg = config.values["preprocessing"]
    # Removing short segments
    if (segment_data["end"] - segment_data["start"]) < cfg["min_segment_length_sec"]:
        return True

    # Removing segments that contain stop_phrases
    matching_stop_phrases = [
        s for s in cfg["stop_phrases"] if s.lower() in segment_data["text"].lower()
    ]
    if matching_stop_phrases:
        # Remove the segment if it contains only a stop phrase
        if max(map(len, matching_stop_phrases)) + cfg["stop_phrase_length_delta"] > len(
            segment_data["text"].strip()
        ):
            return True
    if segment_data["no_speech_prob"] > cfg["no_speech_threshold"]:
        return True

    return False


def segmentize(
    source: pathlib.Path | str,
    whispermodel,
    config: Config,
) -> Generator[Tuple[dict, pathlib.Path]]:
    transcription = whispermodel.transcribe(
        str(source),
        word_timestamps=True,
        language="ru",
    )
    audio = pydub.AudioSegment.from_file(source)
    for segment_data in transcription["segments"]:
        if filter_out(segment_data, config):
            continue
        start = float(segment_data["start"]) * 1000
        end = float(segment_data["end"]) * 1000
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
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
            Metric(
                "duration",
                MetricType.FLOAT,
                duration_seconds,
                "seconds"
            ),
            Metric(
                "sample rate",
                MetricType.INT,
                sample_rate,
                "Hz"
            ),
            Metric(
                "channels",
                MetricType.INT,
                channels,
                None,
            ),
            Metric(
                "bit depth",
                MetricType.INT,
                bit_depth,
                "bits/sample"
            ),
            Metric(
                "max loudness",
                MetricType.FLOAT,
                max_dbfs,
                "dBFS"
            ),
            Metric(
                "rms",
                MetricType.INT,
                rms,
                None,
            ),
            Metric(
                "raw data length",
                MetricType.INT,
                raw_data_length,
                "samples"
            ),
        ],
    )
