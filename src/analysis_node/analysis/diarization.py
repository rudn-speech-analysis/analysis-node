from collections import defaultdict
import logging
import pathlib
import tempfile
import librosa
import numpy as np
import torch
from scipy.io import wavfile

from analysis_node.config import Config


from pyannote.audio import Pipeline, Inference, Model

# Since torch 2.6 serialization behaves differently.
# To avoid errors we need to specify several classes as safe
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio.pipelines.utils.hook import ProgressHook

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from omegaconf.listconfig import ListConfig
torch.serialization.add_safe_globals(
    [
        Specifications,
        Problem,
        Resolution,
        # EarlyStopping,
        # ModelCheckpoint,
        # ListConfig
    ]
)

logger = logging.getLogger(__name__)

class Diarizer:
    def __init__(self, device: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=True,
        ).to(torch.device(device))

    def load_file(self, file: pathlib.Path | str):
        audio_np, sample_rate = librosa.load(file, sr=None, mono=False)

        # # Ensure audio_np is (n_samples, n_channels)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim == 2:
            # librosa loads as (n_channels, n_samples) already
            pass
        else:
            raise ValueError("Unexpected audio dimensions")

        # {'waveform': (channel, time) torch.Tensor, 'sample_rate': int}
        waveform = torch.from_numpy(audio_np).float()
        data = {"waveform": waveform, "sample_rate": sample_rate}
        return data
    
    def get_turns(self, data):
        with ProgressHook() as hook:
            output = self.pipeline(data, hook=hook)
        return output
    
    def get_waveforms(self, data, turns):
        waveform = data["waveform"]
        sample_rate = data["sample_rate"]

        # Group segments by speaker
        speaker_segments = defaultdict(list)
        for segment, speaker in turns.speaker_diarization:
            speaker_segments[speaker].append(segment)
        logger.debug('Total speaker list:', list(speaker_segments.keys()))
        waveform = waveform.T
        total_samples = waveform.shape[0]

        # For each unique speaker, extract and concatenate their audio segments, then save to a WAV file
        for speaker, segments in speaker_segments.items():
            # Silent audio array matching the original
            speaker_audio = np.zeros_like(waveform)
            
            # Sort segments by start time
            segments = sorted(segments, key=lambda seg: seg.start)
            
            for seg in segments:
                # Calculate sample indices (inclusive start, exclusive end)
                start_sample = int(seg.start * sample_rate)
                end_sample = int(seg.end * sample_rate)

                # Ensure bounds
                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)

                speaker_audio[start_sample:end_sample] = waveform[start_sample:end_sample]
            
            # combine all channels into one, if needed
            # waveform has shape (n_channels, n_samples)
            speaker_audio = np.sum(speaker_audio, axis=1)
            yield (speaker, speaker_audio, sample_rate)
    
    def process(self, segment_file: pathlib.Path | str):
        data = self.load_file(segment_file)
        turns = self.get_turns(data)
        for speaker, waveform, sample_rate in self.get_waveforms(data, turns):
            file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wavfile.write(file, sample_rate, waveform)
            logger.info(speaker, 'written to', file.name)
            yield file
