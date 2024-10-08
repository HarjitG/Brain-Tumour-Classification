import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
import numpy as np

class Plotting_functions:
    def __init__(self,class_names, labels_testing=None):
        try:
                
            if not isinstance(class_names,list):
                raise TypeError ("Input must be a list")
            if len(class_names) != 4:
                raise ValueError("The list must contain exactly 4 elements")
            if not all(isinstance(item, str) for item in class_names):
                raise TypeError("All elements in list must be of string format")
            
            self.class_names = class_names
            self.labels_testing = labels_testing

        except (TypeError, ValueError) as e:
            print(f"Error: {e}")
            raise

    def plot_tumour(self,batch_training, label = None):
        try:

            if batch_training[0].shape[1:] != (256,256,1):
                raise ValueError ('Image should be of size (256,256,1)')
            
            # Checking the labels are integers
            if not np.issubdtype(batch_training[1].dtype, np.integer):
                raise ValueError(f"labels should be integers, but got {batch_training[1].dtype}") 

            fig, ax = plt.subplots(ncols = 4, figsize = (20,20))
        
            for idx, img_index in enumerate([1,13,4,5]):


                ax[idx].imshow(batch_training[0][img_index].astype(int))

                #Take default label or assign to custom names
                if label is None:
                    current_label = batch_training[1][img_index]
                else:
                    current_label = label[img_index]

                ax[idx].title.set_text(current_label)

            plt.tight_layout()  # Adjust layout to fit everything nicely
            plt.show()

        except ValueError as e:
            print(f'Error {e}')
            raise

        
    def plot_loss(self,hist):
        try:
            # Checking hist input is valid 
            if not isinstance(hist, tf.keras.callbacks.History):
                raise TypeError (f'Expected Hist object, but received {type(hist)}')
            
            # Checking for relevant keys 
            if ('loss' not in hist.history or 'val_loss' not in hist.history):
                raise ValueError (f'Expected loss and val_loss, and instead got {list(hist.history)}')
            
                                  
            fig = plt.figure()
            plt.plot(hist.history['loss'], color='teal', label='loss')
            plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
            plt.xlabel("Loss", fontsize = 14)
            plt.ylabel("Valiation loss", fontsize = 14)
            fig.suptitle('Loss', fontsize=20)
            plt.legend(loc="upper left")
            plt.show()

        except (TypeError, ValueError) as e:
            print(f"Error: {e}")
            raise


    def plot_accuracy(self,hist):
        try:
            # Checking hist input is valid 
            if not isinstance(hist, tf.keras.callbacks.History):
                raise TypeError (f'Expected Hist object, but received {type(hist)}')
            
            # Checking for relevant keys 
            if ('accuracy' not in hist.history or 'val_accuracy' not in hist.history):
                raise ValueError (f'Expected accuracy and val_accuracy, and instead got {list(hist.history)}')

            fig = plt.figure()
            plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
            plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
            plt.xlabel("Accuracy", fontsize = 14)
            plt.ylabel("Validation accuracy", fontsize = 14)
            fig.suptitle('Accuracy', fontsize=20)
            plt.legend(loc="upper left")
            plt.show()

        except (ValueError, TypeError) as e:
            print (f'Error: {e}')
            raise

    def plot_confusion_matrix(self, y_true, y_pred_binary):

        try:
            # Check if of numpy array
            if not isinstance(y_true, np.ndarray):
                raise TypeError (f'Should be of Numpy array, instead got {type(y_true)}')
            if not isinstance(y_pred_binary, np.ndarray):
                raise TypeError (f'Should be of Numpy array, instead got {type(y_true)}')
            
            # Check dimensionality
            if y_true.ndim != 1:
                raise ValueError (f'Expected a 1D array, instead got array of {y_true.ndim} Dimensions')
            if y_pred_binary.ndim != 1:
                raise ValueError (f'Expected a 1D array, instead got array of {y_pred_binary.ndim} Dimensions')
            
            # Checking number of values matches length of dataset
            if y_true.shape[0] != len(self.labels_testing):
                raise ValueError(f"Expected array of shape (394,), but got {y_true.shape} instead.")
            if y_pred_binary.shape[0] != len(self.labels_testing):
                raise ValueError(f"Expected array of shape (394,), but got {y_pred_binary.shape} instead.")
            
            # Check values are integers
            if not np.issubdtype(y_true.dtype, np.int64):
                raise ValueError(f"Expected integer values but array contains {y_true.dtype} values")
            if not np.issubdtype(y_pred_binary.dtype, np.int64):
                raise ValueError(f"Expected integer values but array contains {y_pred_binary.dtype} values")

            conf = sk_confusion_matrix(y_true, y_pred_binary)
            sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

        except (ValueError, TypeError) as e:
            print (f'Error: {e}')
            raise