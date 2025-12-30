import pathlib
import numpy as np
import torch
import librosa
import logging

from analysis_node.analysis.processors.processor import Processor
from analysis_node.messages import MetricType, Metric, MetricCollection

from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor


logger = logging.getLogger(__name__)


class CatEmotionProcessor(Processor):
    r"""
    Speech emotion classifier.
    Outupts probabilities to for specific emotions.
    """

    MODEL_NAME = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
    FEATURE_EXTRACTOR_NAME = "facebook/hubert-large-ls960-ft"
    REQUIRED_SAMPLING_RATE = 16000

    def __init__(self, device: str = "cpu"):
        super().__init__(self.MODEL_NAME, device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.FEATURE_EXTRACTOR_NAME
        )
        self.model = HubertForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)  # pyright: ignore
        self.model.eval()

        logger.info(f"Done creating {self.__class__.__name__}")

    @torch.no_grad()
    def __call__(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        Analyze raw audio for specific emotion (neutral, angry, positive, sad, other).

        Parameters
        ----------
        audio : np.ndarray
            Raw waveform (1D or 2D).

        Returns
        -------
        np.ndarray
            Predictions.
        """
        audio = np.expand_dims(audio, axis=0)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        return self.model(inputs["input_values"]).logits.cpu().numpy()

    def process(self, segment_file: pathlib.Path | str) -> MetricCollection:
        y, _ = librosa.load(segment_file, sr=self.REQUIRED_SAMPLING_RATE, mono=False)
        vals = self(y)
        metrics = [
            Metric(k, MetricType.FLOAT, v, None)
            for k, v in zip(["neutral", "angry", "positive", "sad", "other"], vals[0])
        ]
        return MetricCollection(self._model_name, metrics)
