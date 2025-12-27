import traceback
import dacite
from kafka import KafkaConsumer, KafkaProducer
from typing import Tuple, Callable
import json
import logging
from dataclasses import asdict

from analysis_node.config import Config
from analysis_node.messages import (
    ErrorMsg,
    KafkaAnalysisRequest,
    KafkaAnalysisResponse,
    AnalysisRequest,
    KafkaMsgOption,
    ProgressMsg,
)
from analysis_node.utils import Generator, NpEncoder
from analysis_node.analysis import MetricsGenerator

logger = logging.getLogger(__name__)


class KafkaSingleTopicProducer(KafkaProducer):
    def __init__(self, topic: str, **kwargs):
        self.topic = topic
        self._producer = KafkaProducer(**kwargs)

    def __getattr__(self, name):
        return getattr(self._producer, name)

    def __setattr__(self, name, value):
        # Handle internal attributes without delegation
        if name in ["_producer", "topic"]:
            super().__setattr__(name, value)
        else:
            # Delegate attribute setting to the wrapped instance
            setattr(self._producer, name, value)

    def __delattr__(self, name):
        delattr(self._producer, name)

    def send(
        self,
        value,
        key=None,
        headers=None,
        partition=None,
        timestamp_ms=None,
    ):
        self._producer.send(
            topic=self.topic,
            value=value,
            key=key,
            headers=headers,
            partition=partition,
            timestamp_ms=timestamp_ms,
        )


def prepare_kafka(config: Config) -> Tuple[KafkaConsumer, KafkaSingleTopicProducer]:
    cfg = config.values["kafka"]

    consumer = KafkaConsumer(
        cfg["topics"]["incoming"],
        bootstrap_servers=cfg["bootstrap_servers"],
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=cfg["group_id"],
        value_deserializer=lambda x: dacite.from_dict(
            data_class=KafkaAnalysisRequest, data=json.loads(x.decode("utf-8"))
        ),
    )
    logger.info("Created Kafka consumer")

    producer = KafkaSingleTopicProducer(
        topic=cfg["topics"]["outgoing"],
        bootstrap_servers=cfg["bootstrap_servers"],
        value_serializer=lambda x: json.dumps(asdict(x), cls=NpEncoder).encode("utf-8"),
    )
    logger.info("Created Kafka producer")

    return consumer, producer


def loop_kafka(
    consumer: KafkaConsumer,
    producer: KafkaSingleTopicProducer,
    processor: Callable[[AnalysisRequest], MetricsGenerator],
):
    def send(id: str, data: KafkaMsgOption):
        producer.send(KafkaAnalysisResponse(id, data))
        producer.flush()

    for message in consumer:
        print(message.value)
        request = message.value
        id = request.id
        logger.info(f"Processing request with id: {id}")

        p = Generator(processor(request.data))
        try:
            for output in p:
                send(id, output)
            send(id, p.value)
            consumer.commit()
            logger.info(f"Done processing request with id: {id}")
        except Exception as e:
            trace = traceback.format_exc()
            send(id, ErrorMsg(str(e), trace))
            logger.error(
                f"Error while processing request with id: {id} "
                f"Exception: {e} "
                f"Trace: {trace}"
            )
