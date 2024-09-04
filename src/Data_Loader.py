import tensorflow as tf

class DataLoader:
    def __init__(self, image_size=(256,256),
                 batch_size = 32,
                 validation_split = 0.2,
                 seed =42):
        try:
            if not (isinstance(image_size,tuple) and len(image_size) == 2 and all(isinstance(i,int) for i in image_size)):
                raise ValueError("Image size must be a tuple of two integers")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Batch size must be positive integer")
            if not (0 < validation_split <1):
                raise ValueError("Validation split must be between 0 and 1")
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer")
            
        except (TypeError, ValueError) as e:
            print(f"Error: {e}")
            raise

        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed

    def Load_Training_data(self, training_path):

        try:
            if not isinstance(training_path, str):
                raise ValueError("Training path should be a string")
        except (ValueError) as e:
            print(f"Error: {e}")
            raise

        # Proceed with data loading

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

        try:
            if not isinstance(testing_path, str):
                raise ValueError("Testing path should be a string")
        except (ValueError) as e:
            print(f"Error: {e}")
            raise

        # Proceed to load testing data

        Testing_data = tf.keras.utils.image_dataset_from_directory(
        testing_path,
        labels='inferred',
        image_size=self.image_size,
        batch_size=self.batch_size,
    )
        return Testing_data
    
    def validate_data(self, dataset):
        try:
            for images,labels in dataset.as_numpy_iterator(): #taking first batch
                print('images', images[1])
                print('labels', labels[1])
                if images.shape[1:] != (256,256,3):
                    raise ValueError ('Image should be of size (256,256,3)')
                if labels.shape[1] != 1:
                    raise ValueError ('Labels should be of length 1' )
        except ValueError as e:
            print(f'Data validation error {e}')
            raise