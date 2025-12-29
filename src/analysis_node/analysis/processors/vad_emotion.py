import pathlib
import numpy as np
import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
import logging

from analysis_node.analysis.processors.processor import Processor
from analysis_node.analysis.processors.utils import ModelHead
from analysis_node.messages import MetricType, Metric, MetricCollection


logger = logging.getLogger(__name__)


class VadEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = ModelHead(config, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


class VadEmotionProcessor(Processor):
    r"""
    Speech emotion classifier.
    Outupts Valency, Arousal and Dominance values.
    """

    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    REQUIRED_SAMPLING_RATE = 16000

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)

        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
        self.model = VadEmotionModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)  # pyright: ignore
        self.model.eval()

        logger.info(f"Created {self.__class__.__name__}")

    @torch.no_grad()
    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        return_embeddings: bool = False,
    ) -> np.ndarray:
        """
        Analyze raw audio for emotion (arousal, dominance, valence) or embeddings.

        Parameters
        ----------
        audio : np.ndarray
            Raw waveform (1D or 2D).
        sampling_rate : int
            Sampling rate of the input audio.
        return_embeddings : bool
            False: Returns [[arousal, dominance, valence]] (~[0,1]).
            True: Returns pooled embeddings [[...]] (shape (1, 1024)).

        Returns
        -------
        np.ndarray
            Predictions or embeddings.
        """

        if audio.ndim == 1:
            # adding batch if not already
            audio = audio[None, :]
        audio = audio.astype(np.float32)

        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,  # pyright: ignore
            return_tensors="pt",  # pyright: ignore
        )

        input_values = inputs["input_values"].to(self.device)

        embeddings, logits = self.model(input_values)

        result = embeddings if return_embeddings else logits

        return result.cpu().numpy()

    def process(self, segment_file: pathlib.Path | str) -> MetricCollection:
        y, sr = librosa.load(segment_file, sr=self.REQUIRED_SAMPLING_RATE, mono=False)
        vals = self(y, int(sr))
        metrics = [
            Metric(k, MetricType.FLOAT, v, None)
            for (k, v) in zip(["arousal", "dominance", "valence"], vals[0])
        ]
        return MetricCollection(self._model_name, metrics)
