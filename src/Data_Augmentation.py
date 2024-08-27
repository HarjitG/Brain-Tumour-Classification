import tensorflow as tf
import random


class Data_Augmentation:
    def __init__(self, threshold = 0.5):
        self.threshold = threshold

    def augment_image(self, x, y):
        # Probability of applying augmentation is 0.5, We don't want to apply augmentation to all images, only some.

        apply_augmentation = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.threshold
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
    
