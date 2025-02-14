from observers.base_observer import BaseObserver

class EvaluationNotifier:
    """Manages observers and notifies them about model evaluation events."""

    def __init__(self):
        """Initializes an empty list of observers."""
        self.observers = []

    def add_observer(self, observer: BaseObserver):
        """
        Adds an observer to the notifier.

        Args:
            observer (BaseObserver): Observer instance to be notified.
        """
        self.observers.append(observer)

    def notify(self, model_name: str, metric_name: str, score: float):
        """
        Notifies all observers about an evaluation event.

        Args:
            model_name (str): Name of the evaluated model.
            metric_name (str): Name of the evaluation metric.
            score (float): Computed score for the model.
        """
        for observer in self.observers:
            observer.update(model_name, metric_name, score)
