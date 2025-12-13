from typing import Generator, Any, Tuple
import whisper
import dacite
import pathlib
import logging

from analysis_node.analysis.processors.processor import Processor
from analysis_node.config import Config
from analysis_node.utils import fetch_to_tmp_file
from analysis_node.messages import (
    AnalysisRequest,
    ChannelMetrics,
    ProgressMsg,
    RecordingMetrics,
    WhisperMetrics,
)
from analysis_node.analysis.processors import EmotionProcessor, AgeGenderProcessor
from analysis_node.analysis.preprocessing import (
    get_audio_metrics,
    split_audio,
    segmentize,
)
from analysis_node.analysis.postprocessing import prepare_channel_metrics_report

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
        self.per_channel_processors: dict[str, Processor] = {
            "age_gender": age_gender_pipeline,
        }
        self.config = config

        logger.info(f"Created {self.__class__.__name__}")

    def _collect_metrics_per_segment(
        self,
        segment_file,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                name: processor.process(segment_file)
                for name, processor in self.per_segment_processors.items()
            },
            {
                name: processor.process(segment_file)
                for name, processor in self.per_channel_processors.items()
            },
        )

    def _collect_metrics_per_channel(
        self,
        channel_file: pathlib.Path | str,
    ) -> Generator[Tuple[dict, dict], None, None]:
        for segment_data, segment_path in segmentize(
            channel_file, self.whisper, self.config
        ):
            per_segment, per_channel = self._collect_metrics_per_segment(segment_path)
            per_segment["whisper"] = dacite.from_dict(
                data_class=WhisperMetrics,
                data=segment_data,
            )
            yield per_segment, per_channel

    def __call__(self, request: AnalysisRequest) -> MetricsGenerator:
        cfg = self.config.values["reporting"]
        source_audio_file = fetch_to_tmp_file(request.download_url)
        audio_metrics = get_audio_metrics(source_audio_file)
        channel_files = split_audio(source_audio_file)

        metrics = []
        for channel in channel_files:
            logger.info(f"Begin processing channel {len(metrics)}")
            raw_channel_metrics = []
            last_progress = 0
            for raw_segment_metrics in self._collect_metrics_per_channel(channel):
                end = raw_segment_metrics[0]["whisper"].end
                percent_done = (end / audio_metrics.duration_seconds) * 100

                if percent_done > last_progress + cfg["progress_delta"]:
                    last_progress = percent_done
                    logger.info(
                        f"Processing channel {len(metrics)}: {round(percent_done)}%"
                    )
                    yield ProgressMsg(percent_done)

                raw_channel_metrics.append(raw_segment_metrics)

            try:
                raw_segment_metrics, raw_channel_metrics = zip(*raw_channel_metrics)
            except:
                raw_segment_metrics, raw_channel_metrics = list(dict()), list(dict())

            logger.info(f"Preparing channel metrics report for channel {len(metrics)}")
            cm = prepare_channel_metrics_report(
                len(metrics),
                raw_segment_metrics,  # pyright: ignore
                raw_channel_metrics,  # pyright: ignore
                audio_metrics.duration_seconds,
            )
            metrics.append(cm)

            yield cm

        return RecordingMetrics(audio_metrics)
