import numpy as np
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


class Cleaner:
    def __init__(self, ModelBuilder=None, pre = None, re = None, acc = None):

        self.ModelBuilder = ModelBuilder
        self.pre = Precision()
        self.re = Recall()
        self.acc = BinaryAccuracy()

    def Testing_data_cleaner(self, Test_data):
        labels_testing = []
        X_test = []

        for batch in Test_data.as_numpy_iterator():
            x_test, y_test = batch
            y_test = [
                np.where(row == 1)[0][0].tolist() for row in y_test
            ]  # Gets the position of the 1 to determine the type of tumor and assigns to a list
            labels_testing.extend(y_test)
            X_test.extend(x_test)

        labels_testing = np.array(labels_testing)  # Full y_test labels transformed
        X_test = np.array(X_test)  # Full X_test data

        return X_test, labels_testing

    def y_predictor(self, X_test, labels_testing):
        y_ = labels_testing  # Our true y values
        yhat_ = []  # Empty list for predicted variables

        yhat = self.ModelBuilder.model.predict(X_test)
        yhat_binary = np.argmax(yhat, axis=1)  # gets value of 1 and position
        yhat_.append(yhat_binary)

        self.pre.update_state(y_, yhat_binary)
        self.re.update_state(y_, yhat_binary)
        self.acc.update_state(y_, yhat_binary)

        print(f"Precision: {self.pre.result().numpy()}")
        print(f"Recall: {self.re.result().numpy()}")
        print(f"Accuracy: {self.acc.result().numpy()}")

        return y_, yhat_binary, self.pre, self.re, self.acc
