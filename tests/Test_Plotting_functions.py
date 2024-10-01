import pytest
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt
import tensorflow as tf
from src.Plotting_functions import Plotting_functions  # Import your class here

# Sample fixture for a valid Plotting_functions instance
@pytest.fixture
def valid_plotting_instance():
    class_names = ["Class1", "Class2", "Class3", "Class4"]
    labels_testing = np.array([0, 1, 2, 3, 1, 0])
    return Plotting_functions(class_names, labels_testing)

# Test plot_confusion_matrix with valid input
def test_plot_confusion_matrix(valid_plotting_instance):
    y_true = np.array([0, 1, 2, 3, 1, 0])
    y_pred = np.array([0, 1, 1, 3, 2, 0])

    with patch("seaborn.heatmap") as mock_heatmap:
        valid_plotting_instance.plot_confusion_matrix(y_true, y_pred)
        mock_heatmap.assert_called_once()  # Ensure heatmap is called once for the plot

# Parametrize test for invalid inputs to plot_confusion_matrix
@pytest.mark.parametrize(
    "y_true, y_pred, error_type",
    [
        (np.array([[1, 2]]), np.array([1, 2]), ValueError),  # Invalid dimension for y_true
        (np.array([1, 2]), np.array([[1, 2]]), ValueError),  # Invalid dimension for y_pred
        (np.array([1.0, 2.0]), np.array([1, 2]), ValueError),  # Non-integer y_true
        (np.array([1, 2]), np.array([1.0, 2.0]), ValueError),  # Non-integer y_pred
    ]
)
def test_invalid_confusion_matrix_input(valid_plotting_instance, y_true, y_pred, error_type):
    with pytest.raises(error_type):
        valid_plotting_instance.plot_confusion_matrix(y_true, y_pred)

# Test the plot_loss with mock data
def test_plot_loss(valid_plotting_instance):
    hist_mock = tf.keras.callbacks.History()
    hist_mock.history = {
        'loss': [0.1, 0.05],
        'val_loss': [0.15, 0.1]
    }

    with patch("matplotlib.pyplot.show") as mock_show:
        valid_plotting_instance.plot_loss(hist_mock)
        mock_show.assert_called_once()  # Ensure the plot is shown

# Test the plot_accuracy with mock data
def test_plot_accuracy(valid_plotting_instance):
    hist_mock = tf.keras.callbacks.History()
    hist_mock.history = {
        'accuracy': [0.8,0.85],
        'val_accuracy': [0.78,0.82]
    }

    with patch("matplotlib.pyplot.show") as mock_show:
        valid_plotting_instance.plot_accuracy(hist_mock)
        mock_show.assert_called_once()  # Ensure the plot is shown
