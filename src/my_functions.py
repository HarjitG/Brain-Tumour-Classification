import tensorflow as tf
import random

def Load_Training_data(training_path):
    Training_data = tf.keras.utils.image_dataset_from_directory(
    training_path,
    labels='inferred',
    image_size=(256, 256), # We are adjusting the images so they become 256x256 images instead.
    batch_size=32,
    validation_split=0.2, # We will utilise 20% of the data from the training data for validation.
    subset='training',
    seed=42
)
    return Training_data

def Load_Validaion_data(training_path):
    Validation_data = tf.keras.utils.image_dataset_from_directory(
    training_path,
    labels='inferred',
    image_size=(256, 256), # We are adjusting the images so they becoem 256x256 images instead.
    batch_size=32,
    validation_split=0.2, # We will utilise 20% of the data from the training data for validation.
    subset='validation',
    seed=42
)
    return Validation_data

def Load_Testing_data(testing_path):
    Testing_data = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    labels='inferred',
    image_size=(256, 256),
    batch_size=32,
)
    return Testing_data

def augment_image(x, y):
    # Probability of applying augmentation is 0.5, We don't want to apply augmentation to all images, only some.

    apply_augmentation = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > 0.5
    if apply_augmentation:
        augmentations = [
            lambda x: tf.image.random_flip_left_right(x),
            lambda x: tf.image.random_flip_up_down(x),
            lambda x: tf.image.rot90(x, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)),
          #  lambda x: tf.image.central_crop(x, central_fraction=tf.clip_by_value(tf.random.uniform(shape=[], minval=0.8, maxval=1.2), 0.0, 1.0)),
        ]

        selected_augmentation = random.choice(augmentations)
        # Apply the selected augmentation to the image
        x = selected_augmentation(x)
    return x, y

# Pre process ia causing issues atm so will apply augmentation seperately and do pre process mapping nmanually

# def preprocess_data(dataset, augment_func):
#         num_classes = 4
#         # Apply data augmentation
#         dataset = dataset.map(augment_image)
#         # Normalize images and one-hot encode labels
#         dataset.map(lambda x, y: (x / 255.0, tf.one_hot(y, num_classes)))
#         print(num_classes)

#         return dataset