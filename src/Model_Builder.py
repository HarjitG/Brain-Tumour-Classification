import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard


class CNNModel:
    def __init__(
        self,
        input_shape=(256, 256, 3),
        num_classes=4,
        Training_data=None,
        Validation_data=None,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.Training_data = Training_data
        self.Validation_data = Validation_data

    def build_model(
        self,
        l2_1=0.002,
        l2_2=0.002,
        l2_3=0.002,
        dropout_rate_1=0.01,
        dropout_rate_2=0.01,
        dropout_rate_3=0.01,
    ):

        self.model = Sequential()

        # First have an input layer, going to have 16 filters, filter is a 3x3, stride of 1
        # Relu activation turns negative values to 0, and preserves positive values
        self.model.add(
            Conv2D(16, (3, 3), 1, activation="relu", input_shape=self.input_shape)
        )
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, (3, 3), 1, activation="relu"))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, (3, 3), 1, activation="relu"))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())  # condense values

        # Fully connected layers
        self.model.add(Dense(64, activation="relu", kernel_regularizer=l2(l2_1)))
        self.model.add(Dropout(rate=dropout_rate_1))

        self.model.add(Dense(256, activation="relu", kernel_regularizer=l2(l2_2)))
        self.model.add(Dropout(rate=dropout_rate_2))

        self.model.add(Dense(256, activation="relu", kernel_regularizer=l2(l2_3)))
        self.model.add(Dropout(rate=dropout_rate_3))

        # Final layer that gives a single output and represets the label
        self.model.add(Dense(4, activation="softmax"))

    def compile(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()

    def train(self, Training_data, Validation_data, epochs=15, log_dir=None, patience=3):
        if log_dir is None:
            raise ValueError("Log_dir not specified")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience)

        # Executing training of model

        hist = self.model.fit(
            x = Training_data,
            validation_data= Validation_data,
            epochs = epochs,
            callbacks=[tensorboard_callback, early_stopping],
        )

        return hist
