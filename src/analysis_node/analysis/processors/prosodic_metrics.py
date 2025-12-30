import logging
import pathlib
import numpy as np
import librosa
from scipy.signal import correlate, find_peaks, stft, medfilt
from typing import Callable

from analysis_node.messages import MetricCollection, Metric, MetricType
from analysis_node.analysis.processors.processor import Processor

logger = logging.getLogger(__name__)

METRIC_EXTRACTORS: list[Callable[[pathlib.Path | str], dict[str, Metric] | Metric]] = (
    list()
)


def metric(func):
    METRIC_EXTRACTORS.append(func)
    return func


class ProsodicProcessor(Processor):
    def __init__(self):
        super().__init__("audio-metrics-processor", "cpu")

        logger.info(f"Done creating {self.__class__.__name__}")

    def process(self, segment_file: pathlib.Path | str) -> MetricCollection:
        outputs = [f(segment_file) for f in METRIC_EXTRACTORS]
        metrics = []
        for output in outputs:
            if isinstance(output, dict):
                for item in output.values():
                    metrics.append(item)
            else:
                metrics.append(output)

        return MetricCollection(self._model_name, metrics, "Prosodic metrics.")


def _load_audio(file_path: pathlib.Path | str) -> tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    Returns sample rate (int) and normalized signal (np.float64 array, mono, [-1,1] range).
    """
    y, fs = librosa.load(str(file_path), sr=None, mono=False, dtype=np.float64)

    if y.ndim > 1:
        y = y[0]

    # Edge case for all-zero signals
    if np.max(np.abs(y)) == 0:
        y = np.zeros_like(y)

    return y, int(fs)


# Helper to frame signal (shared)
def _frame_signal(
    signal: np.ndarray, fs: int, frame_length: float = 0.03, hop_length: float = 0.01
) -> list[np.ndarray]:
    frame_size = int(fs * frame_length)
    hop_size = int(fs * hop_length)
    frames = [
        signal[i : i + frame_size]
        for i in range(0, len(signal) - frame_size + 1, hop_size)
    ]
    return frames


@metric
def extract_f0(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts fundamental frequency (F0/pitch) features.
    Returns mean and std of F0 in Hz.
    Uses autocorrelation for pitch detection.
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    f0_values = []
    for frame in frames:
        if len(frame) == 0:
            continue
        autocorr = correlate(frame, frame, mode="full")[len(frame) :]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr  # Normalize
        peak = np.argmax(autocorr[int(fs / 500) : int(fs / 50)]) + int(
            fs / 500
        )  # Search 50-500 Hz
        f0 = (
            fs / peak if peak > 0 and autocorr[peak] > 0.2 else 0
        )  # Threshold for voiced
        f0_values.append(f0)
    f0_values = np.array(f0_values)
    valid_f0 = f0_values[f0_values > 0]
    if len(valid_f0) == 0:
        mean_f0 = 0.0
        std_f0 = 0.0
    else:
        mean_f0 = np.mean(valid_f0)
        std_f0 = np.std(valid_f0)
    return {
        "mean": Metric(
            "f0_mean", MetricType.FLOAT, mean_f0, "Hz", "Mean fundamental frequency"
        ),
        "std": Metric(
            "f0_std",
            MetricType.FLOAT,
            std_f0,
            "Hz",
            "Standard deviation of fundamental frequency",
        ),
    }


def _lpc(signal, order):
    """
    Compute LPC coefficients using Levinson-Durbin recursion.
    signal: input frame (numpy array)
    order: LPC order (int)
    Returns: LPC coefficients (numpy array of length order + 1, with a[0]=1)
    """
    N = len(signal)
    if N <= order:
        return np.zeros(order + 1)

    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode="full")[N - 1 :] / N
    R = autocorr[: order + 1]

    # Initialize
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = R[0]
    if E == 0:
        return a

    # Levinson-Durbin recursion
    for k in range(1, order + 1):
        lambda_k = -np.dot(a[1:k], R[k - 1 : 0 : -1]) - R[k]
        lambda_k /= E
        a_temp = a.copy()
        a[k] = lambda_k
        a[1:k] += lambda_k * np.flip(a_temp[1:k])
        E *= 1 - lambda_k**2

    return a


@metric
def extract_formant_f1(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts first formant frequency (F1).
    Returns mean and std in Hz (focus on voiced frames via energy threshold).
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(
        signal, fs, frame_length=0.025, hop_length=0.01
    )  # Shorter for formants
    f1_values = []
    for frame in frames:
        if np.mean(frame**2) < 1e-5:  # Skip low-energy
            continue
        coeffs = _lpc(frame, order=12)  # Order ~ fs/1000 + 2-4
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) > 0]  # Upper half
        angles = np.angle(roots)
        freqs = sorted(angles * (fs / (2 * np.pi)))  # Formant freqs
        if len(freqs) >= 1:
            f1 = freqs[0] if 200 < freqs[0] < 900 else 0
            if f1 > 0:
                f1_values.append(f1)
    valid_f1 = np.array(f1_values)
    mean_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0.0
    std_f1 = np.std(valid_f1) if len(valid_f1) > 0 else 0.0
    return {
        "mean": Metric(
            "formant_f1_mean", MetricType.FLOAT, mean_f1, "Hz", "Mean first formant"
        ),
        "std": Metric(
            "formant_f1_std", MetricType.FLOAT, std_f1, "Hz", "Std of first formant"
        ),
    }


@metric
def extract_formant_f2(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts second formant frequency (F2).
    Returns mean and std in Hz (focus on voiced frames via energy threshold).
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(
        signal, fs, frame_length=0.025, hop_length=0.01
    )  # Shorter for formants
    f2_values = []
    for frame in frames:
        if np.mean(frame**2) < 1e-5:  # Skip low-energy
            continue
        coeffs = _lpc(frame, order=12)  # Order ~ fs/1000 + 2-4
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) > 0]  # Upper half
        angles = np.angle(roots)
        freqs = sorted(angles * (fs / (2 * np.pi)))  # Formant freqs
        if len(freqs) >= 2:
            f2 = freqs[1] if 600 < freqs[1] < 2800 else 0
            if f2 > 0:
                f2_values.append(f2)
    valid_f2 = np.array(f2_values)
    mean_f2 = np.mean(valid_f2) if len(valid_f2) > 0 else 0.0
    std_f2 = np.std(valid_f2) if len(valid_f2) > 0 else 0.0
    return {
        "mean": Metric(
            "formant_f2_mean", MetricType.FLOAT, mean_f2, "Hz", "Mean second formant"
        ),
        "std": Metric(
            "formant_f2_std", MetricType.FLOAT, std_f2, "Hz", "Std of second formant"
        ),
    }


@metric
def extract_intensity(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts intensity (loudness) features via RMS energy.
    Returns mean and std of intensity in dB (relative).
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    rms_values = []
    for frame in frames:
        rms = np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0
        rms_values.append(rms)
    rms_values = np.array(rms_values)
    valid_rms = rms_values[rms_values > 0]
    if len(valid_rms) == 0:
        mean_intensity = 0.0
        std_intensity = 0.0
    else:
        mean_intensity = np.mean(20 * np.log10(valid_rms))  # Convert to dB
        std_intensity = np.std(20 * np.log10(valid_rms))
    return {
        "mean": Metric(
            "intensity_mean",
            MetricType.FLOAT,
            mean_intensity,
            "dB",
            "Mean intensity in dB",
        ),
        "std": Metric(
            "intensity_std",
            MetricType.FLOAT,
            std_intensity,
            "dB",
            "Standard deviation of intensity in dB",
        ),
    }


@metric
def extract_zero_crossing_rate(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts zero-crossing rate (ZCR), indicating noisiness or high-frequency content.
    Returns mean and std of ZCR (crossings per second).
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    zcr_values = []
    for frame in frames:
        zcr = (
            np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            if len(frame) > 0
            else 0
        )
        zcr_values.append(zcr * fs)  # Normalize to per second
    zcr_values = np.array(zcr_values)
    mean_zcr = np.mean(zcr_values)
    std_zcr = np.std(zcr_values)
    return {
        "mean": Metric(
            "zcr_mean", MetricType.FLOAT, mean_zcr, "Hz", "Mean zero-crossing rate"
        ),
        "std": Metric(
            "zcr_std",
            MetricType.FLOAT,
            std_zcr,
            "Hz",
            "Standard deviation of zero-crossing rate",
        ),
    }


@metric
def extract_pauses(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts pause-related features within the phrase.
    Detects low-energy segments as pauses (threshold: 0.01 * global RMS, min_pause: 0.1s).
    Returns number of pauses, total pause duration (s), and mean pause duration (s).
    """
    signal, fs = _load_audio(file_path)
    global_rms = np.sqrt(np.mean(signal**2))
    frame_length = 0.03
    hop_length = 0.01
    frame_size = int(fs * frame_length)
    hop_size = int(fs * hop_length)
    energy = np.array(
        [
            np.sqrt(np.mean(signal[i : i + frame_size] ** 2))
            for i in range(0, len(signal) - frame_size, hop_size)
        ]
    )
    is_pause = (
        medfilt((energy < 0.01 * global_rms).astype(float), kernel_size=3) > 0.5
    )  # Smooth
    pause_starts = np.where(np.diff(is_pause.astype(int)) == 1)[0] + 1
    pause_ends = np.where(np.diff(is_pause.astype(int)) == -1)[0] + 1
    if len(pause_starts) > len(pause_ends):
        pause_ends = np.append(pause_ends, len(is_pause))
    if len(pause_ends) > len(pause_starts):
        pause_starts = np.insert(pause_starts, 0, 0)
    pause_durations = (pause_ends - pause_starts) * hop_length
    pause_durations = pause_durations[pause_durations >= 0.1]  # Min pause duration
    num_pauses = len(pause_durations)
    total_duration = np.sum(pause_durations) if num_pauses > 0 else 0.0
    mean_duration = np.mean(pause_durations) if num_pauses > 0 else 0.0
    return {
        "num": Metric(
            "pauses_num", MetricType.INT, num_pauses, None, "Number of pauses"
        ),
        "total_duration": Metric(
            "pauses_total_duration",
            MetricType.FLOAT,
            total_duration,
            "s",
            "Total pause duration",
        ),
        "mean_duration": Metric(
            "pauses_mean_duration",
            MetricType.FLOAT,
            mean_duration,
            "s",
            "Mean pause duration",
        ),
    }


@metric
def extract_speaking_rate(file_path: pathlib.Path | str) -> Metric:
    """
    Extracts speaking rate as estimated syllables per second.
    Syllables approximated by peaks in amplitude envelope (smoothed RMS).
    Returns single rate value (no std, as it's aggregate).
    """
    signal, fs = _load_audio(file_path)
    duration = len(signal) / fs
    if duration == 0:
        return Metric(
            "speaking_rate",
            MetricType.FLOAT,
            0.0,
            "syllables/s",
            "Estimated speaking rate",
        )
    envelope = np.abs(signal)
    kernel_size = int(fs * 0.02)
    if kernel_size % 2 == 0:
        kernel_size += 1
    envelope = medfilt(envelope, kernel_size=kernel_size)  # Smooth ~20ms
    peaks, _ = find_peaks(
        envelope, height=np.mean(envelope) * 0.5, distance=int(fs * 0.1)
    )  # Min distance 100ms
    num_syllables = len(peaks)
    rate = num_syllables / duration
    return Metric(
        "speaking_rate",
        MetricType.FLOAT,
        rate,
        "syllables/s",
        "Estimated speaking rate",
    )


@metric
def extract_jitter(file_path: pathlib.Path | str) -> Metric:
    """
    Extracts jitter (pitch perturbation).
    Returns mean local jitter (%) from F0 values.
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    f0_values = []
    for frame in frames:
        if len(frame) == 0:
            continue
        autocorr = correlate(frame, frame, mode="full")[len(frame) :]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        peak = np.argmax(autocorr[int(fs / 500) : int(fs / 50)]) + int(fs / 500)
        f0 = fs / peak if peak > 0 and autocorr[peak] > 0.2 else 0
        f0_values.append(f0)
    f0_values = np.array(f0_values)
    valid_f0 = f0_values[f0_values > 0]
    if len(valid_f0) < 2:
        jitter = 0.0
    else:
        jitter = (
            np.mean(np.abs(np.diff(valid_f0)) / valid_f0[:-1]) * 100
        )  # Local jitter in %
    return Metric("jitter", MetricType.FLOAT, jitter, "%", "Mean local jitter")


@metric
def extract_shimmer(file_path: pathlib.Path | str) -> Metric:
    """
    Extracts shimmer (amplitude perturbation).
    Returns mean local shimmer (%) from RMS values.
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    rms_values = [np.sqrt(np.mean(frame**2)) for frame in frames if len(frame) > 0]
    rms_values = np.array(rms_values)
    valid_rms = rms_values[rms_values > 0]
    if len(valid_rms) < 2:
        shimmer = 0.0
    else:
        shimmer = (
            np.mean(np.abs(np.diff(valid_rms)) / valid_rms[:-1]) * 100
        )  # Local shimmer in %
    return Metric("shimmer", MetricType.FLOAT, shimmer, "%", "Mean local shimmer")


@metric
def extract_spectral_centroid(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts spectral centroid (brightness indicator).
    Returns mean and std in Hz, computed per STFT frame.
    """
    signal, fs = _load_audio(file_path)
    f, t, Zxx = stft(signal, fs=fs, nperseg=int(fs * 0.03), noverlap=int(fs * 0.01))
    magnitude = np.abs(Zxx)
    centroids = []
    for i in range(magnitude.shape[1]):
        spec = magnitude[:, i]
        if np.sum(spec) == 0:
            continue
        centroid = np.sum(f * spec) / np.sum(spec)
        centroids.append(centroid)
    centroids = np.array(centroids)
    mean_centroid = np.mean(centroids) if len(centroids) > 0 else 0.0
    std_centroid = np.std(centroids) if len(centroids) > 0 else 0.0
    return {
        "mean": Metric(
            "spectral_centroid_mean",
            MetricType.FLOAT,
            mean_centroid,
            "Hz",
            "Mean spectral centroid",
        ),
        "std": Metric(
            "spectral_centroid_std",
            MetricType.FLOAT,
            std_centroid,
            "Hz",
            "Standard deviation of spectral centroid",
        ),
    }


@metric
def extract_spectral_rolloff(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts spectral rolloff (high-frequency energy boundary).
    Returns mean and std in Hz (85% energy threshold).
    """
    signal, fs = _load_audio(file_path)
    f, t, Zxx = stft(signal, fs=fs, nperseg=int(fs * 0.03), noverlap=int(fs * 0.01))
    magnitude = np.abs(Zxx)
    rolloffs = []
    for i in range(magnitude.shape[1]):
        spec = magnitude[:, i]
        if np.sum(spec) == 0:
            continue
        cum_energy = np.cumsum(spec) / np.sum(spec)
        rolloff_idx = np.where(cum_energy >= 0.85)[0][0]
        rolloff = f[rolloff_idx]
        rolloffs.append(rolloff)
    rolloffs = np.array(rolloffs)
    mean_rolloff = np.mean(rolloffs) if len(rolloffs) > 0 else 0.0
    std_rolloff = np.std(rolloffs) if len(rolloffs) > 0 else 0.0
    return {
        "mean": Metric(
            "spectral_rolloff_mean",
            MetricType.FLOAT,
            mean_rolloff,
            "Hz",
            "Mean spectral rolloff (85%)",
        ),
        "std": Metric(
            "spectral_rolloff_std",
            MetricType.FLOAT,
            std_rolloff,
            "Hz",
            "Standard deviation of spectral rolloff",
        ),
    }


@metric
def extract_spectral_entropy(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts spectral entropy (spectral disorder/flatness).
    Returns mean and std (unitless, 0-1 normalized).
    """
    signal, fs = _load_audio(file_path)
    _, _, Zxx = stft(signal, fs=fs, nperseg=int(fs * 0.03), noverlap=int(fs * 0.01))
    magnitude = np.abs(Zxx)
    entropies = []
    for i in range(magnitude.shape[1]):
        spec = magnitude[:, i]
        if np.sum(spec) == 0:
            continue
        prob = spec / np.sum(spec)
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        entropies.append(entropy / np.log2(len(prob)))  # Normalize to 0-1
    entropies = np.array(entropies)
    mean_ent = np.mean(entropies) if len(entropies) > 0 else 0.0
    std_ent = np.std(entropies) if len(entropies) > 0 else 0.0
    return {
        "mean": Metric(
            "spectral_entropy_mean",
            MetricType.FLOAT,
            mean_ent,
            None,
            "Mean normalized spectral entropy",
        ),
        "std": Metric(
            "spectral_entropy_std",
            MetricType.FLOAT,
            std_ent,
            None,
            "Std of normalized spectral entropy",
        ),
    }


@metric
def extract_average_talk_time(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts average talk time features within the phrase.
    Talk segments are non-pause (voiced) parts, detected as inverse of pauses.
    Uses same energy threshold as pauses (0.01 * global RMS, min_talk: 0.1s).
    Returns mean and std of talk segment durations in seconds.
    """
    signal, fs = _load_audio(file_path)
    global_rms = np.sqrt(np.mean(signal**2))
    frame_length = 0.03
    hop_length = 0.01
    frame_size = int(fs * frame_length)
    hop_size = int(fs * hop_length)
    energy = np.array(
        [
            np.sqrt(np.mean(signal[i : i + frame_size] ** 2))
            for i in range(0, len(signal) - frame_size, hop_size)
        ]
    )
    is_pause = (
        medfilt((energy < 0.01 * global_rms).astype(float), kernel_size=3) > 0.5
    )  # Smooth
    is_talk = np.logical_not(is_pause)
    talk_starts = np.where(np.diff(is_talk.astype(int)) == 1)[0] + 1
    talk_ends = np.where(np.diff(is_talk.astype(int)) == -1)[0] + 1
    if len(talk_starts) > len(talk_ends):
        talk_ends = np.append(talk_ends, len(is_talk))
    if len(talk_ends) > len(talk_starts):
        talk_starts = np.insert(talk_starts, 0, 0)
    talk_durations = (talk_ends - talk_starts) * hop_length
    talk_durations = talk_durations[
        talk_durations >= 0.1
    ]  # Min talk duration to filter noise
    if len(talk_durations) == 0:
        mean_talk = 0.0
        std_talk = 0.0
    else:
        mean_talk = np.mean(talk_durations)
        std_talk = np.std(talk_durations) if len(talk_durations) > 1 else 0.0
    return {
        "mean": Metric(
            "talk_time_mean",
            MetricType.FLOAT,
            mean_talk,
            "s",
            "Mean duration of talking segments",
        ),
        "std": Metric(
            "talk_time_std",
            MetricType.FLOAT,
            std_talk,
            "s",
            "Standard deviation of talking segment durations",
        ),
    }


@metric
def extract_hnr(file_path: pathlib.Path | str) -> dict[str, Metric]:
    """
    Extracts Harmonics-to-Noise Ratio (voice quality/clarity).
    Returns mean and std in dB.
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    hnr_values = []
    for frame in frames:
        if len(frame) == 0:
            continue
        autocorr = correlate(frame, frame, mode="full")[len(frame) :]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        peak = np.argmax(autocorr[int(fs / 500) : int(fs / 50)]) + int(fs / 500)
        r_max = autocorr[peak] if peak > 0 else 0
        r_max = np.clip(
            r_max, 0.0, 1.0
        )  # Clamp to [0,1] to handle floating-point precision issues
        if r_max > 0.2:  # Voiced threshold
            denom = max(1 - r_max, 1e-10)  # Ensure denominator is at least 1e-10
            hnr = 10 * np.log10(r_max / denom)
            hnr_values.append(hnr)
    hnr_values = np.array(hnr_values)
    mean_hnr = np.mean(hnr_values) if len(hnr_values) > 0 else 0.0
    std_hnr = np.std(hnr_values) if len(hnr_values) > 0 else 0.0
    return {
        "mean": Metric(
            "hnr_mean",
            MetricType.FLOAT,
            mean_hnr,
            "dB",
            "Mean harmonics-to-noise ratio",
        ),
        "std": Metric(
            "hnr_std", MetricType.FLOAT, std_hnr, "dB", "Standard deviation of HNR"
        ),
    }


@metric
def extract_articulation_rate(file_path: pathlib.Path | str) -> Metric:
    """
    Extracts articulation rate (syllables/s excluding pauses).
    Returns single rate value.
    """
    pauses = extract_pauses(file_path)  # Reuse your function
    total_pause = pauses["total_duration"].value
    speaking_metric = extract_speaking_rate(file_path)
    signal, fs = _load_audio(file_path)
    # Back-calculate approx syllables
    num_syllables = speaking_metric.value * (len(signal) / fs)  # pyright: ignore
    total_dur = len(signal) / fs
    voiced_dur = total_dur - total_pause
    rate = num_syllables / voiced_dur if voiced_dur > 0 else 0.0
    return Metric(
        "articulation_rate",
        MetricType.FLOAT,
        rate,
        "syllables/s",
        "Articulation rate excluding pauses",
    )


@metric
def extract_voiced_ratio(file_path: pathlib.Path | str) -> Metric:
    """
    Extracts ratio of voiced to total frames.
    Returns single float (0-1).
    """
    signal, fs = _load_audio(file_path)
    frames = _frame_signal(signal, fs)
    voiced_count = 0
    for frame in frames:
        autocorr = correlate(frame, frame, mode="full")[len(frame) :]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        peak = np.argmax(autocorr[int(fs / 500) : int(fs / 50)]) + int(fs / 500)
        if peak > 0 and autocorr[peak] > 0.2:
            voiced_count += 1
    ratio = voiced_count / len(frames) if len(frames) > 0 else 0.0
    return Metric(
        "voiced_ratio", MetricType.FLOAT, ratio, None, "Ratio of voiced to total frames"
    )
