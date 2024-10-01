import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock
from src.Model_Builder import CNNModel  # Assuming the class is in a file named cnn_model.py

# Test for the CNNModel initialization
def test_initialization():
    # Test valid initialization
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    assert model.input_shape == (256, 256, 1)
    assert model.num_classes == 4
    assert model.model is None
    
    # Test invalid input shape
    with pytest.raises(ValueError, match="input_shape must be a tuple of three integers"):
        CNNModel(input_shape=(256, 256))  # Missing one dimension

    # Test invalid num_classes
    with pytest.raises(ValueError, match="num_classes must be an integer greater than 1"):
        CNNModel(input_shape=(256, 256, 1), num_classes=1)  # num_classes should be > 1

# Test for building the CNN model
def test_build_model():
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    model.build_model()

    # Check if the model is built and layers are added
    assert model.model is not None
    assert len(model.model.layers) > 0

    # Test invalid L2 regularization value
    with pytest.raises(ValueError, match="l2 value must be a non negative floating number"):
        model.build_model(l2_1=-0.01)  # Invalid L2 regularization

    # Test invalid dropout value
    with pytest.raises(ValueError, match="Value should floating number be between 0 and 1"):
        model.build_model(dropout_rate_1=1.5)  # Invalid dropout rate

# Test for model compilation
def test_compile_model():
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    model.build_model()

    # Test valid compilation
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    assert model.model.optimizer is not None

    # Test invalid optimizer
    with pytest.raises(ValueError, match="Optimizer must be a string"):
        model.compile(optimizer=123)  # Invalid optimizer type

    # Test invalid loss function
    with pytest.raises(ValueError, match="Loss function must be either 'categorical_crossentropy'"):
        model.compile(loss="invalid_loss_function")  # Invalid loss function

# Test for model summary
def test_summary(capsys):
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    model.build_model()

    # Capture the output of the summary
    model.summary()
    captured = capsys.readouterr()
    
    assert "Model: " in captured.out  # Verify that the model summary is printed

# Test for the model training
@patch('tensorflow.keras.callbacks.TensorBoard')
@patch('tensorflow.keras.callbacks.EarlyStopping')
def test_train(mock_tensorboard, mock_early_stopping):
    # Mock the training data
    mock_training_data = MagicMock()
    mock_validation_data = MagicMock()

    # Mock the fit method to avoid actual training
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    model.build_model()

    with patch.object(model.model, 'fit', return_value="Training Success") as mock_fit:
        log_dir = 'logs/'  # Mock log directory
        
        # Call train method with mocked callbacks
        hist = model.train(mock_training_data, mock_validation_data, epochs=10, log_dir=log_dir, 
                           tensorboard_callback=mock_tensorboard.return_value, 
                           early_stopping_callback=mock_early_stopping.return_value)

        # Check if the model's fit method was called with the correct arguments
        mock_fit.assert_called_once_with(
            x=mock_training_data,
            validation_data=mock_validation_data,
            epochs=10,
            callbacks=[mock_tensorboard.return_value, mock_early_stopping.return_value]  # Use .return_value for mock callbacks
        )
        
        print("Hist: ", hist)
        assert hist == "Training Success"  # Check if the fit method returns the expected value



# Test for missing log_dir during training
def test_missing_log_dir():
    model = CNNModel(input_shape=(256, 256, 1), num_classes=4)
    model.build_model()

    # Check if error is raised when log_dir is not provided
    with pytest.raises(ValueError, match="Log_dir not specified"):
        model.train(Training_data=MagicMock(), Validation_data=MagicMock())

