import os
import logging
import pandas as pd

from adapters.json_adapter import JSONAdapter
from utils.load import load_all_ground_truths
from utils.eval import evaluate_by_section, evaluate_by_section_per_section
from reports.report_generator import ReportGenerator

from observers.evaluation_notifier import EvaluationNotifier
from observers.console_logger import ConsoleLogger
from observers.file_logger import FileLogger

class EvaluationPipeline:
    """
    Pipeline for evaluating model outputs and generating reports.

    This class loads ground truth and model output data, computes both global 
    and per-section evaluation metrics, notifies observers about global metric 
    evaluations, and then exports the results to CSV or JSON reports.
    """
    def __init__(self, config, evaluators, report_format: str = None, notifier: EvaluationNotifier = None):
        """
        Initializes the evaluation pipeline.

        Args:
            config (dict): Configuration dictionary.
            evaluators (dict): Dictionary of evaluator instances.
            report_format (str, optional): Report format ("csv" or "json").
                If not provided, the pipeline will use the value from the config.
            notifier (EvaluationNotifier, optional): Notifier for logging evaluation events.
                If not provided, a default notifier is created.
        """
        self.config = config
        self.ground_truth_dir = config.get("ground_truth_dir", "data/ground_truth")
        self.model_outputs_dir = config.get("model_outputs_dir", "data/model_outputs")
        self.evaluators = evaluators
        self.gt_map = load_all_ground_truths(self.ground_truth_dir)
        if not self.gt_map:
            logging.error("No valid ground truth files found. Exiting.")
            raise ValueError("Ground truths not found.")
        self._log_ground_truths()

        self.report_format = (report_format or 
                              config.get("evaluation", {}).get("format", "csv")).lower()


        if notifier is None:
            self.notifier = EvaluationNotifier()
       
            self.notifier.add_observer(ConsoleLogger())
        
            log_file = config.get("log_file")
            if log_file:
                self.notifier.add_observer(FileLogger(log_file))
        else:
            self.notifier = notifier

    def _log_ground_truths(self):
        """Logs information about the loaded ground truth files."""
        logging.info(f"Loading ground truth from directory: {self.ground_truth_dir}")
        for gt_name, data in self.gt_map.items():
            logging.info(f"  {gt_name}: Total Cost = {data['total']}")

    def process_model_file(self, model_file):
        """
        Processes a single model output file.

        This method:
          - Reads and validates the model JSON file.
          - Aggregates ground truth and prediction DataFrames.
          - Computes global and per-section metrics.
          - Notifies observers for each global metric result.

        Args:
            model_file (str): The model output filename.

        Returns:
            tuple: A tuple containing two lists: (global_results, section_results)
                   or None if processing fails.
        """
        model_output_path = os.path.join(self.model_outputs_dir, model_file)
        logging.info(f"Processing model output: {model_output_path}")

        try:
            model_adapter = JSONAdapter(model_output_path)
        except Exception as e:
            logging.error(f"Error loading {model_output_path}: {e}")
            return None

        if "estimate_preds" not in model_adapter.data:
            logging.warning(f"File '{model_file}' is not a valid model output (missing 'estimate_preds').")
            return None

        predictions = model_adapter.to_model_outputs()
        gt_dfs, pred_dfs = [], []
        for prediction in predictions:
            valid_file = prediction.get('valid_file_name')
            if valid_file not in self.gt_map:
                logging.warning(f"Ground truth for '{valid_file}' not found.")
                continue
            gt_dfs.append(self.gt_map[valid_file]['df'])
            pred_dfs.append(prediction['df'])

        if not gt_dfs or not pred_dfs:
            logging.warning(f"Insufficient data for evaluation in '{model_file}'.")
            return None

        aggregated_gt_df = pd.concat(gt_dfs, ignore_index=True)
        aggregated_pred_df = pd.concat(pred_dfs, ignore_index=True)

        global_results = []
        section_results = []

        for metric_name, evaluator in self.evaluators.items():
            score = evaluate_by_section(evaluator, aggregated_gt_df, aggregated_pred_df)
            global_results.append({
                'model_file': model_file,
                'metric': metric_name,
                'score': score
            })
            self.notifier.notify(model_file, metric_name, score)

        for metric_name, evaluator in self.evaluators.items():
            section_scores = evaluate_by_section_per_section(evaluator, aggregated_gt_df, aggregated_pred_df)
            for section, score in section_scores.items():
                section_results.append({
                    'model_file': model_file,
                    'sectionName': section,
                    'metric': metric_name,
                    'score': score
                })
                self.notifier.notify(model_file, f"{metric_name} [{section}]", score)

        return global_results, section_results

    def run(self):
        """
        Runs the evaluation pipeline on all model output files and generates reports.
        
        Aggregates results from all files and then uses a report generator to export the results.
        """
        all_global_results = []
        all_section_results = []

        for model_file in os.listdir(self.model_outputs_dir):
            results = self.process_model_file(model_file)
            if results is None:
                continue
            global_results, section_results = results
            all_global_results.extend(global_results)
            all_section_results.extend(section_results)

        global_df = pd.DataFrame(all_global_results)
        section_df = pd.DataFrame(all_section_results)

        eval_config = self.config.get("evaluation", {})
        output_path = eval_config.get("output_path", "reports/evaluation_report")
        base_output = os.path.splitext(output_path)[0]

        global_output = f"{base_output}_global.{self.report_format}"
        section_output = f"{base_output}_by_section.{self.report_format}"

        report_generator = ReportGenerator(self.report_format)
        report_generator.generate(global_df, global_output)
        report_generator.generate(section_df, section_output)

        logging.info(f"Reports successfully exported: {global_output} and {section_output}")
