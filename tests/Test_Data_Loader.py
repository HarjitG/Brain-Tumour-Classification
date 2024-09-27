import pytest
from src.Data_Loader import DataLoader  # Replace with actual module name where DataLoader is defined
import numpy as np
import tensorflow as tf

# Fixture to initialize DataLoader class
@pytest.fixture
def data_loader():
    return DataLoader(image_size=(256, 256), batch_size=32, validation_split=0.2, seed=42)

# Test the DataLoader initialization and input validation
def test_dataloader_init(data_loader):
    # Test correct initialization
    assert data_loader.image_size == (256, 256)
    assert data_loader.batch_size == 32
    assert data_loader.validation_split == 0.2
    assert data_loader.seed == 42

    # Test invalid image size
    with pytest.raises(ValueError):
        DataLoader(image_size=(256,))  # Should raise an error for incorrect size

    # Test invalid batch size
    with pytest.raises(ValueError):
        DataLoader(batch_size=-1)  # Should raise an error for negative batch size

    # Test invalid validation split
    with pytest.raises(ValueError):
        DataLoader(validation_split=1.5)  # Should raise an error for invalid split

    # Test invalid seed
    with pytest.raises(ValueError):
        DataLoader(seed="invalid_seed")  # Should raise an error for invalid seed

# Test loading training data with invalid path
def test_load_training_data_invalid_path(data_loader):
    with pytest.raises(ValueError):
        data_loader.Load_Training_data(1234)  # Non-string path

# Test loading testing data with invalid path
def test_load_testing_data_invalid_path(data_loader):
    with pytest.raises(ValueError):
        data_loader.Load_Testing_data(1234)  # Non-string path

# Test data validation function
def test_validate_data(data_loader):
    # Simulate dummy data with correct shape and dtype
    dummy_images = np.random.randint(0, 255, size=(32, 256, 256, 1), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 4, size=(32,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(32)

    # This should pass since the shape and dtype are correct
    data_loader.validate_data(dataset)

    # Simulate incorrect image shape
    incorrect_images = np.random.randint(0, 255, size=(32, 128, 128, 3), dtype=np.uint8)
    dataset_incorrect_shape = tf.data.Dataset.from_tensor_slices((incorrect_images, dummy_labels)).batch(32)
    with pytest.raises(ValueError, match="Image should be of size"):
        data_loader.validate_data(dataset_incorrect_shape)

    # Simulate incorrect label dtype
    incorrect_labels = np.random.random(size=(32,))  # Floating-point labels
    dataset_incorrect_labels = tf.data.Dataset.from_tensor_slices((dummy_images, incorrect_labels)).batch(32)
    with pytest.raises(ValueError, match="labels should be integers"):
        data_loader.validate_data(dataset_incorrect_labels)

    # Simulate incorrect label shape
    incorrect_label_shape = np.random.randint(0, 4, size=(32, 4))  # Labels with wrong shape
    dataset_incorrect_label_shape = tf.data.Dataset.from_tensor_slices((dummy_images, incorrect_label_shape)).batch(32)
    with pytest.raises(ValueError, match="Labels should be a 1D array"):
        data_loader.validate_data(dataset_incorrect_label_shape)
