import tensorflow as tf
import mlflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard


class CNNModel:
    def __init__(
        self,
        input_shape=(256, 256, 1),
        num_classes=4,
        Training_data=None,
        Validation_data=None
        ):
        try:
            # Error handling for input_shape
            if not (isinstance(input_shape, tuple) and len(input_shape) == 3 and all(isinstance(i, int) for i in input_shape)):
                raise ValueError("input_shape must be a tuple of three integers (height, width, channels).")
            
            # Error handling for num_classes
            if not (isinstance(num_classes, int) and num_classes != 1):
                raise ValueError("num_classes must be an integer greater than 1.")
            # Training and validating data is checked in the a cell right before by another method so no need to check here.


            self.input_shape = input_shape
            self.num_classes = num_classes
            self.model = None
            self.Training_data = Training_data
            self.Validation_data = Validation_data
            
        except (ValueError) as e:
            print(f"Initialization Error: {e}")
            raise


    def build_model(
        self,
        l2_1=0.005,
        l2_2=0.01,
        l2_3=0.005,
        dropout_rate_1=0.005,
        dropout_rate_2=0.01,
        dropout_rate_3=0.005,
    ):
    
        try:
            # Error handling for L2 regularisation.
            if not (isinstance(l2_1, (int, float)) and l2_1 >= 0):
                raise ValueError("l2 value must be a non negative floating number")
            if not (isinstance(l2_2, (int, float)) and l2_2 >= 0):
                raise ValueError("l2 value must be a non negative floating number")
            if not (isinstance(l2_3, (int, float)) and l2_3 >= 0):
                raise ValueError("l2 value must be a non negative floating number")
            
            # Error handling for dropout rate
            if not (isinstance(dropout_rate_1, (int, float)) and 0 <= dropout_rate_1 <= 1):
                raise ValueError("Value should floating number be between 0 and 1")
            if not (isinstance(dropout_rate_2, (int, float)) and 0 <= dropout_rate_2 <= 1):
                raise ValueError("Value should floating number be between 0 and 1")
            if not (isinstance(dropout_rate_3, (int, float)) and 0 <= dropout_rate_3 <= 1):
                raise ValueError("Value should floating number be between 0 and 1")
            
            mlflow.log_param("l2_1", l2_1)
            mlflow.log_param("l2_2", l2_2)
            mlflow.log_param("l2_3", l2_3)
            mlflow.log_param("dropout_rate_1", dropout_rate_1)
            mlflow.log_param("dropout_rate_2", dropout_rate_2)
            mlflow.log_param("dropout_rate_3", dropout_rate_3)
            
        except ValueError as e:
            print (f"Error: {e}")
            raise

        

        self.model = Sequential()

        # First have an input layer, going to have 16 filters, filter is a 3x3, stride of 1
        # Relu activation turns negative values to 0, and preserves positive values
        self.model.add(
            Conv2D(16, (3, 3), 1, activation="relu", input_shape=self.input_shape)
        )
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(16, (3, 3), 1, activation="relu"))
        self.model.add(MaxPooling2D())

        self.model.add(Conv2D(64, (3, 3), 1, activation="relu"))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())  # condense values

        # Fully connected layers
        self.model.add(Dense(64, activation="relu", kernel_regularizer=l2(l2_1)))
        self.model.add(Dropout(rate=dropout_rate_1))

        self.model.add(Dense(256, activation="relu", kernel_regularizer=l2(l2_2)))
        self.model.add(Dropout(rate=dropout_rate_2))

        self.model.add(Dense(64, activation="relu", kernel_regularizer=l2(l2_3)))
        self.model.add(Dropout(rate=dropout_rate_3))

        # Final layer that gives a single output and represets the label
        self.model.add(Dense(4, activation="softmax"))

    def compile(
        self, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]):
        try:
            if not isinstance(optimizer, str):
                raise ValueError("Optimizer must be a string indicating the type of optimizer.")
            if loss not in ['categorical_crossentropy', 'sparse_categorical_crossentropy']:
                raise ValueError("Loss function must be either 'categorical_crossentropy' or 'sparse_categorical_crossentropy'.")
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            print('Model compiled successfully')

            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("loss_function", loss)

        except (ValueError, TypeError) as e:
            print(f'Error: {e}')
            raise

        

    def summary(self):
        self.model.summary()
    
    def train(self, Training_data, Validation_data, epochs=15, log_dir=None, patience=3, tensorboard_callback=None, early_stopping_callback=None):
        if log_dir is None:
            raise ValueError("Log_dir not specified")
        
        # Use provided callbacks if given, otherwise default to real callbacks
        tensorboard_callback = tensorboard_callback or tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        early_stopping_callback = early_stopping_callback or tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)

        # Executing training of the model
        hist = self.model.fit(
            x=Training_data,
            validation_data=Validation_data,
            epochs=epochs,
            callbacks=[tensorboard_callback, early_stopping_callback],
        )

        # Log final training and validation metrics
        mlflow.log_metric("final_train_loss", hist.history['loss'][-1])
        mlflow.log_metric("final_val_loss", hist.history['val_loss'][-1])
        mlflow.log_metric("final_train_accuracy", hist.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", hist.history['val_accuracy'][-1])

        return hist
