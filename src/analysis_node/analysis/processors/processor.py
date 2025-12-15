from collections import defaultdict, Counter
import pathlib

from analysis_node.messages import MetricType, Metric, MetricCollection


class Processor:
    def __init__(self, model_name: str):
        self._model_name = model_name

    def process(self, segment_file: pathlib.Path | str) -> MetricCollection:
        raise NotImplemented


class AggregateProcessor(Processor):
    def aggregate(self, metrics: list[MetricCollection]) -> MetricCollection:
        provider = self._model_name
        collection_description = "Aggregate metrics"
        if not metrics:
            return MetricCollection(provider, [], collection_description)

        # Collect values and types for each metric name across all collections
        metric_values = defaultdict(list)
        metric_types = dict()
        metric_descriptions = dict()  # To store first non-None description per metric

        for mc in metrics:
            for m in mc.metrics:
                # Check for type consistency
                if m.name in metric_types:
                    if metric_types[m.name] != m.type:
                        raise ValueError(
                            f"Inconsistent type for metric '{m.name}': expected {metric_types[m.name]}, got {m.type}"
                        )
                else:
                    metric_types[m.name] = m.type

                # Collect value
                metric_values[m.name].append(m.value)

                # Collect description if not already set
                if m.name not in metric_descriptions and m.description is not None:
                    metric_descriptions[m.name] = m.description

        # Compute aggregated metrics
        aggregated_metrics = []
        for name, values in metric_values.items():
            if not values:
                continue  # Skip if no values (though unlikely)

            typ = metric_types[name]
            description = metric_descriptions.get(name)  # May be None

            if typ in (MetricType.INT, MetricType.FLOAT):
                # Convert to float for mean calculation
                numeric_values = [float(v) for v in values]
                mean_value = sum(numeric_values) / len(numeric_values)
                agg_type = MetricType.FLOAT
                agg_value = mean_value

            elif typ == MetricType.BOOL:
                # Convert to 1.0/0.0 and compute mean proportion
                numeric_values = [1.0 if v else 0.0 for v in values]
                mean_value = sum(numeric_values) / len(numeric_values)
                agg_type = MetricType.FLOAT
                agg_value = mean_value

            elif typ == MetricType.STR:
                # Use most frequent string (mode)
                counter = Counter(values)
                most_common = counter.most_common(1)[0][0]
                agg_type = MetricType.STR
                agg_value = most_common

            else:
                raise ValueError(f"Unsupported metric type '{typ}' for aggregation")

            aggregated_metrics.append(
                Metric(
                    name=name, type=agg_type, value=agg_value, description=description
                )
            )

        return MetricCollection(
            provider=provider,
            metrics=aggregated_metrics,
            description=collection_description,
        )
