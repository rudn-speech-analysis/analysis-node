import yaml
import jsonschema
import pathlib
from dataclasses import dataclass


@dataclass
class Config:
    values: dict


SCHEMA = {
    "type": "object",
    "properties": {
        "server": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
        "kafka": {
            "type": "object",
            "properties": {
                "max_poll_interval_ms": {"type": "integer"},
                "bootstrap_servers": {"type": "string"},
                "group_id": {"type": "string"},
                "topics": {
                    "type": "object",
                    "properties": {
                        "incoming": {"type": "string"},
                        "outgoing": {"type": "string"},
                    },
                    "required": ["incoming", "outgoing"],
                },
            },
            "required": ["bootstrap_servers", "group_id", "topics"],
        },
        "models": {
            "type": "object",
            "properties": {
                "hf_token": {"type": "string"},
                "device": {"type": "string"},
                "whisper": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "oneOf": [
                                {"const": "turbo"},
                                {"const": "large"},
                            ],
                        },
                        "lang": {"type": "string"},
                    },
                    "required": ["model", "lang"],
                },
                # "emotion2vec": {"type": "object", "properties": {}},
                # "wav2vec2_emotion": {"type": "object", "properties": {}},
                "wav2vec2_age_gender": {
                    "type": "object",
                    "properties": {
                        "num_layers": {
                            "type": "integer",
                            "oneOf": [
                                {"const": 6},
                                {"const": 24},
                            ],
                        },
                    },
                    "required": ["num_layers"],
                },
                "anomaly_detection": {
                    "type": "object",
                    "properties": {
                        "model_file": {"type": "string"},
                    },
                    "required": ["model_file"],
                },
            },
            "required": [
                "device",
                "whisper",
                # "emotion2vec",
                # "wav2vec2_emotion",
                "wav2vec2_age_gender",
                "anomaly_detection",
            ],
        },
        "preprocessing": {
            "type": "object",
            "properties": {
                "min_segment_length_sec": {"type": "number", "minimum": 0},
                "min_segment_distance_sec": {
                    "oneOf": [
                        {"type": "number", "minimum": 0},
                        {"type": "boolean", "const": False},
                    ]
                },
                "no_speech_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "stop_phrases": {"type": "array", "items": {"type": "string"}},
                "stop_phrase_length_delta": {"type": "integer"},
            },
            "required": [
                "min_segment_length_sec",
                "min_segment_distance_sec",
                "stop_phrases",
                "stop_phrase_length_delta",
                "no_speech_threshold",
            ],
        },
        # "postprocessing": {
        #     "type": "object",
        #     "properties": {
        #         "no_speech_threshold": {
        #             "type": "number",
        #             "minimum": 0,
        #             "maximum": 1,
        #         },
        #     },
        #     "required": [
        #         "no_speech_threshold",
        #     ],
        # },
        "reporting": {
            "type": "object",
            "properties": {
                "progress_delta": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": [
                "progress_delta",
            ],
        },
        "logging": {
            "type": "object",
        },
    },
    "required": [
        "kafka",
        "models",
        "preprocessing",
        # "postprocessing",
        "reporting",
        "logging",
    ],
    "additionalProperties": False,
}


def load_yaml_config(path: pathlib.Path | str) -> dict:
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")


def validate_config(config, schema):
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:  # pyright: ignore
        raise ValueError(f"Config  validation failed: {e.message}")


def prepare_config(path: pathlib.Path | str | dict) -> Config:
    if isinstance(path, dict):
        config = path
    else:
        config = load_yaml_config(path)
    jsonschema.validate(instance=config, schema=SCHEMA)
    return Config(values=config)
