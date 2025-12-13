from typing import Any
from dataclasses import fields, is_dataclass
import statistics

from analysis_node.messages import ChannelMetrics, SegmentMetrics, WhisperMetrics


def list_dict_to_dict_list(data: list[dict]) -> dict[str, list]:
    """
    list[dict[str, Any]] -> dict[str, list[Any]]
    list[dataclass[str, num]] -> dict[str, list[num]]
    """

    if not data:
        return dict()
    if is_dataclass(data[0]):
        field_names = [field.name for field in fields(data[0])]
        return {name: [getattr(obj, name) for obj in data] for name in field_names}
    else:
        return {key: [d[key] for d in data] for key in data[0]}


def get_mean_channel_metrics(raw_channel_metrics: list[dict[str, Any]]):
    raw_channel_metrics_list = list_dict_to_dict_list(raw_channel_metrics)

    avg_channel_metrics = dict()
    for k, vals in raw_channel_metrics_list.items():
        vals = list_dict_to_dict_list(vals)

        # [dict[str, list[Any]]] -> dict[str, num]
        mean = dict()
        for vk, vv in vals.items():
            try:
                mean[vk] = statistics.mean(vv)
            except statistics.StatisticsError:
                mean[vk] = 0.0

        avg_channel_metrics[k] = mean

    return avg_channel_metrics


def get_talk_percent(
    whisper_data: list[WhisperMetrics],
    total_duration_sec: float,
) -> float:
    if total_duration_sec <= 0:
        return 0.0

    speech_duration = 0.0
    for segment in whisper_data:
        speech_duration += segment.end - segment.start

    percentage = (speech_duration / total_duration_sec) * 100.0
    return round(percentage, 2)


def prepare_channel_metrics_report(
    channel_idx: int,
    raw_segment_metrics: list[dict[str, SegmentMetrics]],
    raw_channel_metrics: list[dict[str, Any]],
    total_audio_duration_sec: float,
) -> ChannelMetrics:
    segment_metrics = list_dict_to_dict_list(raw_segment_metrics)
    mean_channel_metrics = get_mean_channel_metrics(raw_channel_metrics)
    if "whisper" in segment_metrics.keys():
        talk_percent = get_talk_percent(
            segment_metrics["whisper"],  # pyright: ignore
            total_audio_duration_sec,
        )
    else:
        talk_percent = 0.0

    return ChannelMetrics(
        idx=channel_idx,
        segments=raw_segment_metrics,  # pyright: ignore
        talk_percent=talk_percent,
        **mean_channel_metrics,
    )
