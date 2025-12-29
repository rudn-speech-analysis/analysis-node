from typing import Generator, Tuple
import whisper
import dacite
import pathlib
import logging

from analysis_node.analysis.processors import AggregateProcessor, Processor
from analysis_node.config import Config
from analysis_node.utils import fetch_to_tmp_file, list_dict_to_dict_list
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
    def __init__(self, device, config: Config):
        cfg = config.values["models"]

        login(cfg["hf_token"])

        self.diarizer = Diarizer(device)

        self.whisper = whisper.load_model(cfg["whisper"]["model"]).to(device)

        self.per_segment_processors: dict[str, Processor] = {
            "vad_emotion": VadEmotionProcessor(device),
            "cat_emotion": CatEmotionProcessor(device),
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
                    logger.debug(
                        "processing segment %s with %s", segment_file, proc_name
                    )
                    data[proc_name] = processor.process(segment_file)
                except Exception as e:
                    logger.error(
                        "Error while processing %s on segment %s: %s",
                        proc_name,
                        segment_file,
                        e,
                    )
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
        None,
    ]:
        for segment_data, segment_path in segmentize(
            channel_file, self.whisper, self.config
        ):
            per_segment, per_channel = self._collect_metrics_per_segment(segment_path)
            yield per_segment, per_channel, segment_data

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

        source_audio_file = fetch_to_tmp_file(request.download_url)
        audio_metrics = get_audio_metrics(source_audio_file)

        yield ProgressMsg(0, None, "Splitting audio.")

        # channel_files = list(split_audio(source_audio_file))
        channel_files = self.diarizer.process(source_audio_file)

        metrics = []
        for channel_file in channel_files:
            with channel_file:
                channel = pathlib.Path(channel_file.name)
                channel_idx = len(metrics)
                last_progress = 0

                logger.info(
                    f"Begin processing channel {channel_idx} from file {channel.name}"
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
                for (
                    segment_metrics,
                    channel_metrics,
                    whisper_data,
                ) in self._collect_metrics_per_channel(channel):
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

                logger.debug("about to send:", cm)
                logger.debug(f"{channel_metrics_log=}")
                logger.debug(f"{segment_metrics_log=}")
                logger.debug(f"{whisper_data_log=}")

                yield cm

        return RecordingMetrics([audio_metrics])
