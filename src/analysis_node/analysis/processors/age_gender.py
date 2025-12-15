import numpy as np
import torch
import pathlib
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from scipy.io import wavfile
import logging

from analysis_node.analysis.processors.processor import AggregateProcessor
from analysis_node.analysis.processors.utils import ModelHead, resample_to
from analysis_node.messages import MetricType, Metric, MetricCollection

logger = logging.getLogger(__name__)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


class AgeGenderProcessor(AggregateProcessor):
    SMALL_MODEL_NAME = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    LARGE_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    REQUIRED_SAMPLING_RATE = 16000

    def __init__(self, size: str | int = "small", device: str = "cpu"):
        if isinstance(size, str):
            if size not in {"small", "large"}:
                ValueError('Please select one of "large", "small" models sizes.')
            model_name = (
                self.SMALL_MODEL_NAME if size == "small" else self.LARGE_MODEL_NAME
            )
        elif isinstance(size, int):
            if size not in {6, 24}:
                ValueError('Please select one of "24", "6" models sizes.')
            model_name = self.SMALL_MODEL_NAME if size == 6 else self.LARGE_MODEL_NAME
        else:
            ValueError(
                "Invalid model size specification. "
                'Please use either strings: ["small", "large"] '
                "or numbers: [6, 24] to specify the model size."
            )

        super().__init__(model_name)
        self.device = torch.device(device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name)
        self.model.to(self.device)
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
            Raw waveform (1D or 2D with batch).
        sampling_rate : int
            Sampling rate of the input audio.
        return_embeddings : bool
            False: Returns [[age, female, male, child]] (~[0,1]).
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

        if sampling_rate != self.REQUIRED_SAMPLING_RATE:
            audio = resample_to(audio, sampling_rate, self.REQUIRED_SAMPLING_RATE)
            sampling_rate = self.REQUIRED_SAMPLING_RATE

        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].to(self.device)

        embeddings, age, gender = self.model(input_values)

        result = embeddings if return_embeddings else torch.hstack([age, gender])

        return result.cpu().numpy()

    def process(self, segment_file: pathlib.Path | str) -> MetricCollection:
        sample_rate, waveform = wavfile.read(segment_file)
        vals = self(waveform, sample_rate)
        metrics = [
            Metric(k, MetricType.INT if k == "age" else MetricType.FLOAT, v)
            for (k, v) in zip(["age", "female", "male", "child"], vals[0])
        ]
        return MetricCollection(self._model_name, metrics)
