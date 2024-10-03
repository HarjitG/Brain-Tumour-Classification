import unittest
from unittest.mock import patch
import tensorflow as tf
import numpy as np
from src.Data_Augmentation import Data_Augmentation

class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        # Setting up the Data_Augmentation object
        self.augmentor = Data_Augmentation(threshold=0.5)
        self.image = tf.random.uniform((256, 256, 1), minval=0, maxval=255, dtype=tf.float32)  # Example random image
        self.label = tf.constant(1)  # Dummy label

    @patch('src.Data_Augmentation.tf.random.uniform', return_value=0.6)
    def test_augment_image_applied(self, mock_uniform):
        augmented_image, label = self.augmentor.augment_image(self.image, self.label)

        # Check that the image was augmented
        self.assertFalse(np.array_equal(self.image.numpy(), augmented_image.numpy()))
        self.assertEqual(label.numpy(), 1)

        # Assert that tf.random.uniform was called exactly once
        mock_uniform.assert_called_once()


    @patch('src.Data_Augmentation.tf.random.uniform', return_value=0.4)  # Mocking to ensure augmentation does not happen
    def test_augment_image_not_applied(self, mock_uniform):
        # Augmentation should not be applied when tf.random.uniform < threshold (0.5 in this case)

        augmented_image, label = self.augmentor.augment_image(self.image, self.label)

        # Check that the image was not augmented (should be equal to the original image)
        self.assertTrue(np.array_equal(self.image.numpy(), augmented_image.numpy()))
        self.assertEqual(label.numpy(), 1)  # Ensure label is not changed

    def test_invalid_threshold(self):
        # Testing invalid threshold values
        with self.assertRaises(ValueError):
            Data_Augmentation(threshold=1.5)  # Above valid range

        with self.assertRaises(ValueError):
            Data_Augmentation(threshold=-0.1)  # Below valid range


if __name__ == '__main__':
    unittest.main()
