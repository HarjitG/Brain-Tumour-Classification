import tensorflow as tf

class DataLoader:
    def __init__(self, image_size=(256,256),
                 batch_size = 32,
                 validation_split = 0.2,
                 seed =42):
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed

    def Load_Training_data(self, training_path):
        Training_data = tf.keras.utils.image_dataset_from_directory(
        training_path,
        labels='inferred',
        image_size=self.image_size, # We are adjusting the images so they become 256x256 images instead.
        batch_size=self.batch_size,
        validation_split=self.validation_split, # We will utilise 20% of the data from the training data for validation.
        subset='training',
        seed=42
    )
        return Training_data

    def Load_Validaion_data(self,training_path):
        Validation_data = tf.keras.utils.image_dataset_from_directory(
        training_path,
        labels='inferred',
        image_size=self.image_size, # We are adjusting the images so they become 256x256 images instead.
        batch_size=self.batch_size,
        validation_split=self.validation_split, # We will utilise 20% of the data from the training data for validation.
        subset='validation',
        seed=42
    )
        return Validation_data

    def Load_Testing_data(self,testing_path):
        Testing_data = tf.keras.utils.image_dataset_from_directory(
        testing_path,
        labels='inferred',
        image_size=self.image_size,
        batch_size=self.batch_size,
    )
        return Testing_data