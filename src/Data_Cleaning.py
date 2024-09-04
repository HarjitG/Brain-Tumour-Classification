import numpy as np
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


class Cleaner:
    def __init__(self, ModelBuilder=None, pre = None, re = None, acc = None):

        self.ModelBuilder = ModelBuilder
        self.pre = Precision() if pre is None else pre
        self.re = Recall() if re is None else re
        self.acc = BinaryAccuracy() if acc is None else acc

    def Testing_data_cleaner(self, Test_data):
        try:
            if Test_data is None:
                raise ValueError("Testing data cannot be empty")
            if not hasattr(Test_data, 'as_numpy_iterator'):
                raise ValueError("Testing data must be Tensorflow object")
        except (ValueError) as e:
           print(f'Error: {e}')
           raise

        labels_testing = []
        X_test = []

        for batch in Test_data.as_numpy_iterator():
            x_test, y_test = batch
            # Error checking
            print(x_test.shape)
            print(y_test.shape)
            print('------------')

            print(x_test[0])
            print(y_test[0])





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
