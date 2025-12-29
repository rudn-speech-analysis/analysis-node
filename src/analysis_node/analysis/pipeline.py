from typing import Generator, Tuple
import numpy as np
import whisper
import traceback
import pathlib
import logging
import librosa

from analysis_node.analysis.processors import AggregateProcessor, Processor
from analysis_node.analysis.processors.prosodic_metrics import ProsodicProcessor
from analysis_node.analysis.wer import compute_wer
from analysis_node.config import Config
from analysis_node.utils import GeneratorReturnCatcher, fetch_to_tmp_file, list_dict_to_dict_list
from analysis_node.messages import (
    Metric,
    MetricCollection,
    AnalysisRequest,
    ChannelMetrics,
    MetricType,
    ProgressMsg,
    RecordingMetrics,
    Segment,
)
from analysis_node.analysis.processors import (
    VadEmotionProcessor,
    CatEmotionProcessor,
    AgeGenderProcessor,
)
from analysis_node.analysis.preprocessing import (
    get_audio_metrics,
    split_audio,
    segmentize,
)
from analysis_node.analysis.postprocessing import get_talk_percent, WhisperMetrics

from analysis_node.analysis.diarization import Diarizer
from huggingface_hub import login

logger = logging.getLogger(__name__)


MetricsGenerator = Generator[ChannelMetrics | ProgressMsg, None, RecordingMetrics]


class AnalysisPipeline:
    def __init__(self, config: Config):
        cfg = config.values["models"]
        device = cfg["device"]

        if "hf_token" in cfg and cfg["hf_token"]:
            login(cfg["hf_token"])
        else:
            login()

        self.diarizer = Diarizer(device)

        self.whisper = whisper.load_model(cfg["whisper"]["model"]).to(device)

        self.per_segment_processors: dict[str, Processor] = {
            "vad_emotion": VadEmotionProcessor(device),
            "cat_emotion": CatEmotionProcessor(device),
            "prosodic": ProsodicProcessor(),
        }
        self.per_channel_processors: dict[str, AggregateProcessor] = {
            "age_gender": AgeGenderProcessor(
                cfg["wav2vec2_age_gender"]["num_layers"],
                device,
            ),
        }
        self.config = config

        logger.info(f"Created {self.__class__.__name__}")

    def _collect_metrics_per_segment(
        self,
        segment_file: pathlib.Path | str,
    ) -> Tuple[dict[str, MetricCollection], dict[str, MetricCollection]]:
        def process(
            processors: dict[str, Processor] | dict[str, AggregateProcessor],
        ) -> dict[str, MetricCollection]:
            data = {}
            for proc_name, processor in processors.items():
                try:
                    logger.debug(f"processing segment {segment_file} with {proc_name}")
                    data[proc_name] = processor.process(segment_file)
                except Exception as e:
                    logger.error(
                        f"Error while processing {proc_name} on segment {segment_file}: {e}"
                    )
                    traceback.print_exc()
                    continue
            return data

        return (
            process(self.per_segment_processors),
            process(self.per_channel_processors),
        )

    def _collect_metrics_per_channel(
        self,
        channel_file: pathlib.Path | str,
    ) -> Generator[
        Tuple[dict[str, MetricCollection], dict[str, MetricCollection], WhisperMetrics],
        None,
        list[WhisperMetrics],
    ]:
        """
        Yields some metrics per each merged segment.
        Returns raw Whisper metrics per unmerged segment.
        """
        gen = GeneratorReturnCatcher(segmentize(
            channel_file, self.whisper, self.config
        ))

        for segment_data, segment_path in gen:
            per_segment, per_channel = self._collect_metrics_per_segment(segment_path)
            yield per_segment, per_channel, segment_data
        return gen.value

    def _prepare_channel_metrics_report(
        self,
        channel_idx: int,
        segment_metrics_log: list[dict[str, MetricCollection]],
        channel_metrics_log: list[dict[str, MetricCollection]],
        whisper_data_log: list[WhisperMetrics],
        total_audio_duration_sec: float,
    ) -> ChannelMetrics:
        channel_metrics = list_dict_to_dict_list(channel_metrics_log)
        aggregated_channel_metrics = [
            self.per_channel_processors[k].aggregate(v)
            for k, v in channel_metrics.items()
        ]
        segments = [
            Segment(w.start, w.end, w.text, list(sm.values()))
            for sm, w in zip(segment_metrics_log, whisper_data_log)
        ]

        talk_percent = get_talk_percent(
            whisper_data_log,
            total_audio_duration_sec,
        )
        aggregated_channel_metrics.append(
            MetricCollection(
                provider="whisper",
                description="Aggregate metrics from Whisper",
                metrics=[
                    Metric(
                        name="talk percent",
                        type=MetricType.FLOAT,
                        value=talk_percent,
                        unit="%",
                        description="Fraction of time that this channel was talking",
                    )
                ],
            )
        )

        return ChannelMetrics(
            channel_idx,
            segments,
            aggregated_channel_metrics,
        )

    def __call__(self, request: AnalysisRequest) -> MetricsGenerator:
        cfg = self.config.values["reporting"]

        yield ProgressMsg(0, None, "Loading audio.")

        source_audio_file_obj = fetch_to_tmp_file(request.download_url)
        with source_audio_file_obj:
            source_audio_file = pathlib.Path(source_audio_file_obj.name)
            audio_metrics = get_audio_metrics(source_audio_file)

            yield ProgressMsg(0, None, "Splitting audio.")

            y, sr = librosa.load(source_audio_file, sr=None, mono=False)

            if y.ndim != 2 or y.shape[0] != 2:
                channel_files = self.diarizer.process(y, sr)
            else:
                channel_files = split_audio(y, sr)

            unmerged_whisper_metrics: list[WhisperMetrics] = []

            metrics = []
            for channel_file in channel_files:
                with channel_file:
                    channel = pathlib.Path(channel_file.name)
                    channel_idx = len(metrics)
                    last_progress = 0

                    logger.info(
                        f"Begin processing channel {channel_idx} from file {channel}"
                    )
                    yield ProgressMsg(
                        last_progress, channel_idx, "Begin processing channel."
                    )

                    segment_metrics_log = list()
                    channel_metrics_log = list()
                    whisper_data_log = list()

                    percent_done = 0
                    duration_seconds: float = audio_metrics[
                        "duration"
                    ].value  # pyright: ignore

                    gen = GeneratorReturnCatcher(
                        self._collect_metrics_per_channel(channel)
                    )

                    for (
                        segment_metrics,
                        channel_metrics,
                        whisper_data,
                    ) in gen:
                        end = whisper_data.end
                        percent_done = round((end / duration_seconds) * 100)

                        if percent_done > last_progress + cfg["progress_delta"]:
                            last_progress = percent_done
                            logger.info(
                                f"Processing channel {channel_idx}: {round(percent_done)}%"
                            )
                            yield ProgressMsg(percent_done, channel_idx, None)

                        segment_metrics_log.append(segment_metrics)
                        channel_metrics_log.append(channel_metrics)
                        whisper_data_log.append(whisper_data)

                    unmerged_whisper_metrics.extend(gen.value)

                    logger.info(
                        f"Preparing channel metrics report for channel {len(metrics)}"
                    )
                    yield ProgressMsg(
                        percent_done, channel_idx, "Preparing channel metrics report."
                    )
                    cm = self._prepare_channel_metrics_report(
                        len(metrics),
                        segment_metrics_log,
                        channel_metrics_log,
                        whisper_data_log,
                        duration_seconds,
                    )
                    metrics.append(cm)

                    yield cm

            unmerged_whisper_metrics.sort(key=lambda m: m.start)
            all_text = ' '.join([i.text for i in unmerged_whisper_metrics])

            wer_metric_collection = MetricCollection("internal/word_error_rate", [], "Word error rate calculation")

            if not request.transcript_url:
                return RecordingMetrics([audio_metrics])

            yield ProgressMsg(
                0, None, "Downloading user-provided transcript."
            )
            try:
                transcript_file = fetch_to_tmp_file(request.transcript_url)
            except Exception as e:
                logger.error(f"Error downloading transcript: {e}")
                wer_metric_collection.metrics.append(Metric("wer_error", MetricType.STR, "error downloading user-provided transcript: " + str(e), unit=None, description="Details for the error while calculating word-error rate"))
                return RecordingMetrics([audio_metrics, wer_metric_collection])

            logger.info(f"Transcript file: {transcript_file.name}")
            yield ProgressMsg(10, None, "Reading transcript.")
            try:
                with open(transcript_file.name, "r") as f:
                    transcript_text = f.read()
                    logger.info("Transcript text: " + transcript_text)
                    logger.info("Transcript length: " + str(len(transcript_text)))
            except Exception as e:
                logger.error(f"Error reading transcript: {e}")
                wer_metric_collection.metrics.append(Metric("wer_error", MetricType.STR, "error reading user-provided transcript as text: " + str(e), unit=None, description="Details for the error while calculating word-error rate"))
                return RecordingMetrics([audio_metrics, wer_metric_collection])

            yield ProgressMsg(20, None, "Calculating word error rate.")
            logging.info(f"Transcript: {transcript_text}")
            logging.info(f"Audio: {all_text}")
            wer = compute_wer(transcript_text, all_text)
            if np.isnan(wer) or np.isinf(wer):
                wer_metric_collection.metrics.append(Metric("wer_error", MetricType.STR, "The calculated word-error rate was infinite. To prevent numeric overflow, it has been replaced with -1.", unit=None, description="Details for the error while calculating word-error rate"))
                wer = -1
            wer_metric_collection.metrics.append(Metric("wer", MetricType.FLOAT, wer, None, "Word error rate"))

            return RecordingMetrics([audio_metrics, wer_metric_collection])