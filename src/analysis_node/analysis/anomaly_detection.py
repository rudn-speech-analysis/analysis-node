import joblib
import pandas as pd
import logging
from sklearn.pipeline import Pipeline

from analysis_node.config import Config
from analysis_node.messages import Metric, MetricCollection, MetricType

logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    # excluded because of high correlation with the rest of the data
    EXCLUDE_METRICS = [
        "spectral_centroid_std",
        "hnr_std",
        "spectral_rolloff_mean",
        "zcr_mean",
    ]

    # These aren't useful for anomaly detection
    NON_ANOMALOUS = [
        "age_gender",
    ]

    def __init__(self, config: Config):
        model_file = config.values["models"]["anomaly_detection"]["model_file"]
        self.pipeline: Pipeline = joblib.load(model_file)

        logger.info(f"Done creating {self.__class__.__name__}")

    def __call__(self, data: dict[str, MetricCollection]) -> MetricCollection:
        anomalous = [v for k, v in data.items() if k not in self.NON_ANOMALOUS]
        metrics = {
            m.name: [m.value]
            for collection in anomalous
            for m in collection.metrics
            if m.name not in self.EXCLUDE_METRICS
        }
        df = pd.DataFrame.from_dict(metrics)
        df = df[sorted(df.columns)]

        anomaly_score = self.pipeline.decision_function(df)[0]

        return MetricCollection(
            provider="anomaly-detection-model",
            description="Anomaly metrics from IsolationForest model",
            metrics=[
                Metric("anomaly_score", MetricType.FLOAT, anomaly_score, unit=None)
            ],
        )
