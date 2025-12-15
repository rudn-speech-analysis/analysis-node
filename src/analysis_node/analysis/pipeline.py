from typing import Generator, Tuple
import whisper
import dacite
import pathlib
import logging

from analysis_node.analysis.processors.processor import AggregateProcessor, Processor
from analysis_node.config import Config
from analysis_node.utils import fetch_to_tmp_file, list_dict_to_dict_list
from analysis_node.messages import (
    MetricCollection,
    AnalysisRequest,
    ChannelMetrics,
    ProgressMsg,
    RecordingMetrics,
    Segment,
)
from analysis_node.analysis.processors import EmotionProcessor, AgeGenderProcessor
from analysis_node.analysis.preprocessing import (
    get_audio_metrics,
    split_audio,
    segmentize,
)
from analysis_node.analysis.postprocessing import get_talk_percent, WhisperMetrics

logger = logging.getLogger(__name__)


MetricsGenerator = Generator[ChannelMetrics | ProgressMsg, None, RecordingMetrics]


class AnalysisPipeline:
    def __init__(self, device, config: Config):
        cfg = config.values["models"]

        self.whisper = whisper.load_model(cfg["whisper"]["model"]).to(device)
        emotion_pipeline = EmotionProcessor(device)
        age_gender_pipeline = AgeGenderProcessor(
            cfg["wav2vec2_age_gender"]["num_layers"],
            device,
        )

        self.per_segment_processors: dict[str, Processor] = {
            "emotion": emotion_pipeline,
        }
        self.per_channel_processors: dict[str, AggregateProcessor] = {
            "age_gender": age_gender_pipeline,
        }
        self.config = config

        logger.info(f"Created {self.__class__.__name__}")

    def _collect_metrics_per_segment(
        self,
        segment_file,
    ) -> Tuple[dict[str, MetricCollection], dict[str, MetricCollection]]:
        def process(processors: dict[str, Processor]) -> dict[str, MetricCollection]:
            return {
                proc_name: processor.process(segment_file)
                for proc_name, processor in processors.items()
            }

        return (
            process(self.per_segment_processors),
            process(self.per_channel_processors),  # pyright: ignore
        )

    def _collect_metrics_per_channel(
        self,
        channel_file: pathlib.Path | str,
    ) -> Generator[
        Tuple[dict[str, MetricCollection], dict[str, MetricCollection], WhisperMetrics],
        None,
        None,
    ]:
        for segment_data, segment_path in segmentize(
            channel_file, self.whisper, self.config
        ):
            per_segment, per_channel = self._collect_metrics_per_segment(segment_path)
            whisper_data = dacite.from_dict(
                data_class=WhisperMetrics,
                data=segment_data,
            )
            yield per_segment, per_channel, whisper_data

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

        return ChannelMetrics(
            channel_idx,
            talk_percent,
            segments,
            aggregated_channel_metrics,
        )

    def __call__(self, request: AnalysisRequest) -> MetricsGenerator:
        cfg = self.config.values["reporting"]
        source_audio_file = fetch_to_tmp_file(request.download_url)
        audio_metrics = get_audio_metrics(source_audio_file)
        channel_files = split_audio(source_audio_file)

        metrics = []
        for channel in channel_files:
            logger.info(f"Begin processing channel {len(metrics)}")

            segment_metrics_log = list()
            channel_metrics_log = list()
            whisper_data_log = list()

            last_progress = 0
            for (
                segment_metrics,
                channel_metrics,
                whisper_data,
            ) in self._collect_metrics_per_channel(channel):
                end = whisper_data.end
                duration_seconds: float = audio_metrics[
                    "duration seconds"
                ].value  # pyright: ignore
                percent_done: int = round((end / duration_seconds) * 100)

                if percent_done > last_progress + cfg["progress_delta"]:
                    last_progress = percent_done
                    logger.info(
                        f"Processing channel {len(metrics)}: {round(percent_done)}%"
                    )
                    yield ProgressMsg(percent_done)

                segment_metrics_log.append(segment_metrics)
                channel_metrics_log.append(channel_metrics)
                whisper_data_log.append(whisper_data)

            logger.info(f"Preparing channel metrics report for channel {len(metrics)}")
            cm = self._prepare_channel_metrics_report(
                len(metrics),
                segment_metrics_log,
                channel_metrics_log,
                whisper_data_log,
                duration_seconds,
            )
            metrics.append(cm)

            yield cm

        return RecordingMetrics([audio_metrics])
