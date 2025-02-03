import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from keras.src.saving import load_model
from sklearn.metrics import confusion_matrix
from src.utils import root_dir as root


class ModelPlotter:
    def __init__(self, model_path, test_dataset, model_name, history_path=None):
        """
        Initializes the ModelPlotter class.

        :param model_path: Path to the saved model file (.keras)
        :param test_dataset: Test dataset for evaluation
        :param model_name: Name of the model (e.g., "ResNet50")
        :param history_path: Path to the training history Pickle file (optional)
        """
        self.model = load_model(model_path)
        self.test_dataset = test_dataset
        self.class_names = test_dataset.class_names
        self.model_name = model_name
        self.history = self._load_history(history_path) if history_path else None

    def _load_history(self, history_path):
        """
        Loads the training history from a Pickle file.

        :param history_path: Path to the Pickle file
        :return: Training history dictionary
        """
        with open(history_path, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        return history

    def plot_confusion_matrix(self):
        """
        Plots and saves the confusion matrix.
        """
        y_true = []
        y_pred = []

        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))

        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {self.model_name}")

        plot_dir = os.path.join(root, "Plots", self.model_name)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f'{plot_dir}/confusion_matrix.png')
        plt.show()

    def plot_training_metrics(self):
        """
        Plots and saves the loss, accuracy, and mAP from training history.
        """
        if not self.history:
            print("Training history not available. Please provide a valid history path.")
            return

        metrics = ["loss", "accuracy", "mAP"]
        for metric in metrics:
            if metric in self.history:
                plt.figure(figsize=(8, 6))
                plt.plot(self.history[metric], label=f'Training {metric}')
                plt.plot(self.history[f'val_{metric}'], label=f'Validation {metric}')
                plt.title(f"{metric.capitalize()} Over Epochs")
                plt.xlabel("Epochs")
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)

                plot_dir = os.path.join(root, "Plots", self.model_name)
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(f'{plot_dir}/{metric}.png')
                plt.show()
            else:
                print(f"Metric '{metric}' not found in training history.")
