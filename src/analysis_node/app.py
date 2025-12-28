import argparse
import logging.config

from analysis_node.analysis import AnalysisPipeline
from analysis_node.config import prepare_config
from analysis_node.message_queue import prepare_kafka, loop_kafka


def main():
    parser = argparse.ArgumentParser(prog="analysis-node")
    parser.add_argument(
        "--config",
        "-c",
        help="Path to the config file.",
        required=True,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Pytorch device for the models to use.",
        required=True,
    )

    args = parser.parse_args()
    config = prepare_config(args.config)
    logging.config.dictConfig(config.values["logging"])

    analysis_pipeline = AnalysisPipeline(args.device, config)
    kafka_consumer, kafka_producer = prepare_kafka(config)

    loop_kafka(kafka_consumer, kafka_producer, analysis_pipeline)


if __name__ == "__main__":
    main()
