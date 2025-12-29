from analysis_node.analysis.preprocessing import WhisperMetrics


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
