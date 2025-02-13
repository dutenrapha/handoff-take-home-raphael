import os
import yaml
import logging

from evaluators.mae_evaluator import MAE
from evaluators.mre_evaluator import MRE
from evaluators.mape_evaluator import MAPE
from evaluators.evaluator_pipeline import EvaluationPipeline
from observers.evaluation_notifier import EvaluationNotifier
from observers.console_logger import ConsoleLogger
from observers.file_logger import FileLogger

EVALUATOR_MAPPING = {
    "MAE": MAE,
    "MRE": MRE,
    "MAPE": MAPE,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from: {CONFIG_PATH}")
    else:
        config = {}
        logging.info("Configuration file not found. Using default values.")

    metric_names = config.get("evaluation", {}).get("metrics", [])

    evaluators = {}
    for metric in metric_names:
        metric_upper = metric.upper()
        if metric_upper in EVALUATOR_MAPPING:
            evaluators[metric_upper] = EVALUATOR_MAPPING[metric_upper]()
        else:
            logging.warning(f"Unsupported metric specified in config: {metric}")

    report_format = config.get("evaluation", {}).get("format", "csv")

    notifier = EvaluationNotifier()
    notifier.add_observer(ConsoleLogger())
    log_file = config.get("log_file")
    if log_file:
        notifier.add_observer(FileLogger(log_file))

    pipeline = EvaluationPipeline(config, evaluators, report_format=report_format, notifier=notifier)
    pipeline.run()
