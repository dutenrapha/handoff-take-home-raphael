# AI-ML Evaluation Framework

## Overview
This project is an evaluation framework designed to assess the performance of Large Language Models (LLMs) in generating construction cost estimates. The framework is built with modularity, scalability, and CI/CD compatibility in mind, allowing easy integration and extensibility.

## Features
- **CI/CD Friendly**: Modular and scalable architecture.
- **Evaluation Metrics**: Supports MAE, MAPE, MRE, and Asymmetric Loss.
- **Extensibility**: Easily add new evaluation metrics and data adapters.
- **Automated Reports**: Generates reports in CSV or JSON formats.
- **Logging Support**: Optional logging for evaluation tracking.

---

## Prerequisites
Before running the project, ensure you have the following installed:

- **Docker** (>= 20.10)
- **Docker Compose** (>= 1.29.2)
- **Make** (Linux/macOS: pre-installed, Windows: install via [choco](https://chocolatey.org/packages/make))

## Installation
Clone the repository and navigate to the project directory:

```sh
git git@github.com:dutenrapha/handoff-take-home-raphael.git
cd handoff-take-home-raphael.git
```

Build the necessary Docker images:

```sh
make build
```

---

## Running the Evaluation
### 1. Configure the `config.yaml` file
Modify `config.yaml` to set the required fields:

```yaml
ground_truth_dir: "data/ground_truth"
model_outputs_dir: "data/model_outputs"
evaluation:
  metrics:
    - MAE
    - MAPE
    - MRE
    - ASYMMETRIC
  format: "json"  # Options: "json" or "csv"
  output_path: "reports/20250214_evaluation_report"
log_file: "logs/evaluation.log"  # Optional, omit if logging is not needed
```

**Mandatory Fields:**
- `ground_truth_dir`: Path to ground truth data.
- `model_outputs_dir`: Path to model-generated estimates.
- `format`: Output format (`json` or `csv`).
- `output_path`: Path where evaluation reports will be saved.

At least one of the following metrics must be specified in `metrics`:
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **MRE**: Mean Relative Error
- **ASYMMETRIC**: Custom asymmetric loss function

### 2. Run the Evaluation Pipeline
Start the evaluation using Docker Compose:

```sh
make evaluation
```

Or manually:

```sh
docker-compose up ai_ml_evaluator
```

The results will be saved in `reports/` in the specified format (`json` or `csv`).

---

## Running the UI
A Streamlit app is available to visualize the evaluation results. **Before running the UI, make sure the evaluation has been executed first.** Then, start the UI with:

```sh
make ui
```

This will launch the UI at:

```
http://0.0.0.0:8501
```

### UI Explanation
The UI provides an interactive dashboard for analyzing the evaluation results, as shown in the image:

1. **Metric Selection**: A dropdown at the top allows selecting the metric to visualize.
2. **Global Table**: Displays the overall evaluation scores of different models, ranking them accordingly.
3. **Table by Section**: Breaks down the evaluation scores by construction categories such as Demolition, Framing, and Concrete, allowing for deeper insights into model performance.
4. **Best Model by Section**: Highlights the best-performing model for each section based on the selected metric.

This interface provides a clear comparison between the model-generated cost estimates and the ground truth, allowing users to identify strengths and weaknesses in the model outputs.

---

## Cleaning Up
To stop and remove containers, run:

```sh
make clean
```

To remove all images and generated files:

```sh
make fclean
```

---

## Extending the Framework
### Adding a Custom Evaluator
To add a new evaluation metric, create a new evaluator in `evaluators/` and inherit from `BaseEvaluator`:

```python
from evaluators.base_evaluator import BaseEvaluator
import numpy as np

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, ground_truth, predictions):
        # Implement your metric calculation
        return np.mean((ground_truth - predictions) ** 2)
```

Register the new evaluator in `main.py` under `EVALUATOR_MAPPING`:

```python
EVALUATOR_MAPPING = {
    "MAE": MAE,
    "MRE": MRE,
    "MAPE": MAPE,
    "ASYMMETRIC": AsymmetricLoss,
    "CUSTOM": CustomEvaluator,  # Add this line
}
```

### Adding a New Data Adapter
To support additional data formats (e.g., XML, Parquet), create a new adapter in `adapters/` that extends `BaseAdapter`:

```python
from adapters.base_adapter import BaseAdapter
import pandas as pd

class XMLAdapter(BaseAdapter):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._parse_xml()

    def _parse_xml(self):
        # Implement XML parsing logic here
        pass

    def to_dataframe(self):
        return pd.DataFrame(self.data)
```

Modify `load.py` to include the new adapter based on file extension.

---

## Logging and Debugging
Logs are saved in `logs/evaluation.log` if configured in `config.yaml`.
To view logs in real-time:

```sh
tail -f logs/evaluation.log
```

For debugging, rebuild and run with:

```sh
docker-compose up --build
```

---

## Project Structure
```
.
├── app
│   ├── adapters/        # Data format handlers (e.g., JSON)
│   ├── evaluators/      # Metric evaluation functions
│   ├── observers/       # Event-driven logging
│   ├── reports/         # Report generation modules
│   ├── utils/           # Helper functions
│   ├── data/            # Ground truth and model outputs
│   ├── logs/            # Log files
│   ├── config.yaml      # Configuration file
│   ├── Dockerfile       # Docker setup for evaluation service
│   ├── main.py          # Entry point for evaluation pipeline
├── ui/                  # Streamlit UI for visualization
├── docker-compose.yml   # Docker Compose setup
├── Makefile             # Makefile for automation
└── README.md            # Documentation
```



## Author
Developed by Raphael de Moraes Dutenkefer

