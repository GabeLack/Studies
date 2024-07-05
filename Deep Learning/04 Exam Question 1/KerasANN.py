import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import math
import re
import zipfile
import pickle
from validation import Validation as val

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score,
                             classification_report, ConfusionMatrixDisplay)

from validation import Validation as val

# The test.ipynb file in this directory showcases the class and what I intended from it.

class Keras_Model():
    """A class to create and train Keras Sequential models for both classification
    and regression tasks.

    Parameters:
    - out_activation (str): Activation function for the output layer.
    - layers (tuple): Tuple specifying the number of neurons in each hidden layer. 
      Positive integers represent fully connected layers, 
      while negative floats represent dropout layers.
    - activation (str): Activation function for the hidden layers.

    Attributes:
    - model (Sequential): Keras Sequential model.
    - layers_ (tuple): Tuple specifying the number of neurons in each hidden layer.
    - activation_ (str): Activation function for the hidden layers.
    - out_activation_ (str): Activation function for the output layer.
    - classes_ (None or list): Classes for classification tasks.
    - y_title (None or str): Target column name.
    - scaler (None or MinMaxScaler): Scaler object for feature scaling.
    - X_train (None or np.ndarray): Scaled training features.
    - X_test (None or np.ndarray): Scaled testing features.
    - y_train (None or np.ndarray): Training labels.
    - y_test (None or np.ndarray): Testing labels.
    - y_pred (None or np.ndarray): Predicted labels.
    - losses (None or pd.DataFrame): Losses during training.
    - X_columns_ (None or pd.Index): Column names of input features.
    - y_classes_ (None or list): Classes for classification tasks.
    - best_loss_ (None or float): Best loss during training.
    - last_loss_ (None or float): Last loss during training.
    """
    def __init__(self, out_activation, layers:tuple=(100), activation='relu'):
        """Initialize the Keras_Model object.

        Parameters:
        - out_activation (str): Activation function for the output layer.
        - layers (tuple): Tuple specifying the number of neurons in each hidden layer. Negative float between 0-1 
          for dropout percentage.
        - activation (str): Activation function for the hidden layers.
        """
        # Validation for layers.
        if not val.validate_tuple_ints_floats(layers):
            raise ValueError("Layers error")
        # Thought of putting a validation of activations with a class variable, but seemed fine to
        # use a try-except block to catch valid activations.
        # Example input for layers (100,-0.5,200,-0.5,100,1).
        self.out_activation_ = out_activation
        self.layers_ = layers
        self.activation_ = activation

        # Define attributes in None states for tracking.
        self.model = None
        self.classes_ = None
        self.y_title = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.losses = None

        self.X_columns_ = None
        self.y_classes_ = None
        self.best_loss_ = None
        self.last_loss_ = None

        # Create layers.
        self.model = Sequential()
        for i in layers:
            if i > 0:
                try: # Use error managemant in Dense to raise error further.
                    self.model.add(Dense(i, activation=activation))
                except Exception as e:
                    raise ValueError(f"An error occurred (layer): {e}")
            elif -1 < i < 0: # dropout layer, cannot be i<=-1.
                self.model.add(Dropout(-i))
            else:
                raise ValueError(f"error in layer construction, invalid layer value: {i}\n" +\
                    f"value {i} should be a positive integer or negative float between -1 and 0.")
        # Possibility of adding convolutional layers into the layers list as a tuple?
        # Such as Example input ((16,(4,4)(2,2)),100,-0.5,200,-5,100,1),
        # Which would be filters=16, kernel_size = (4,4), and pooling_size = (2,2)

        try: 
            self.model.add(Dense(layers[-1], activation=out_activation))
        except Exception as e:
            raise ValueError(f'An error occurred (final layer/out activation function): {e}')

    def fit(self, dataset_path, target_col, test_size=0.25, # data inputs.
            monitor:str='val_loss', patience=20, mode:str='auto', # earlystopping inputs.
            batch_size=None, epochs=None, optimizer='adam', # fit inputs.
            loss='auto', metrics:list=None, verbose=1,
            use_multiprocessing=False):
        """Train the model using the provided dataset path and settings.

        Parameters:
        - dataset_path (str): Path to the CSV dataset file.
        - target_col (str): Name of the target column in the dataset.
        - test_size (float): Proportion of the dataset to include in the test split.
        - monitor (str): Quantity to be monitored for early stopping.
        - patience (int): Number of epochs with no improvement after which training will be stopped.
        - mode (str): One of {"auto", "min", "max"}. In "min" mode, training stops when the quantity
          monitored has stopped decreasing. In "max" mode, it will stop when the quantity monitored
          has stopped increasing. In "auto" mode, the direction is automatically inferred.
        - batch_size (int): Number of samples per gradient update.
        - epochs (int): Number of epochs to train the model.
        - optimizer (str or tf.keras.optimizers.Optimizer): Name of the optimizer or opt. instance.
        - loss (str or tf.keras.losses.Loss): Name of the loss function or loss function instance.
        - metrics (list of str): List of metrics to be evaluated by the model during training.
        - verbose (int): Verbosity mode (0, 1, or 2).
        - use_multiprocessing (bool): Whether to use multiprocessing for data preprocessing.
        """
        # try to make df.
        try:
            # Data cleaned and ready to process in csv dataset form.
            df = pd.read_csv(dataset_path)
            # Check for non-encoded categorical columns and missing values
            if df.isna().sum().sum() > 0:
                raise ValueError("Missing values detected in the dataset.")

            # Check for non-encoded categorical columns
            categorical_columns = df.drop(target_col, axis=1).select_dtypes(include=['object']).columns.tolist()
            if categorical_columns:
                raise ValueError(f"Non-encoded categorical columns detected: {', '.join(categorical_columns)}. "
                                "Ensure all categorical columns are properly encoded in the csv before fitting the model.")
        except FileNotFoundError as e:
            raise ValueError(f"An error occured (dataset_path): {e}")
        except ValueError as e:
            raise ValueError(f"An error occured: {e}")

        # Validate target column
        if not target_col in df.columns:
            raise ValueError("An error occured: target_col not found in dataframe.")
        X = df.drop(target_col, axis=1)
        self.X_columns_ = X.columns # Only save column names so less memory is used for later plots.

        y = df[target_col]
        self.y_title = y.name.title() # Same with title, less memory usage.
        # Change if 'object' dtype to get_dummies.
        if y.dtype == 'object':
            self.y_classes_ = y.unique()
            y = pd.get_dummies(y, dtype='int')

        # If epochs and batch_size are empty, put in determined values
        epochs_2, batch_size_2 = self._epoch_batch()
        epochs = epochs or epochs_2
        batch_size = batch_size or batch_size_2

        # Validate reasonable test_size.
        if not (isinstance(test_size, float) and 0 < test_size < 0.5):
            raise ValueError("An error occured: test_size bad outside of range 0 to 0.5.")
        # Train-test split and scaling.
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X.values, y.values, test_size=test_size, random_state=101)
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        # To choose loss automatically based on last layer.
        if loss == 'auto':
            if self.model.layers[-1].activation == 'softmax':
                loss = 'categorical_crossentropy'
            elif self.model.layers[-1].activation == 'sigmoid':
                loss = 'binary_crossentropy'
            else: # continuous
                loss = 'mse'

        # Use internal error raising of compile, EarlyStopping, and fit to raise error to user.
        # I don't know if this counts as 'validation' really?
        try:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            early_stop = EarlyStopping(monitor = monitor, mode = mode,
                                    patience = patience, verbose = verbose,
                                    restore_best_weights = True)

            model_loss = self.model.fit(x=self.X_train, y=self.y_train,
                                        validation_data=(self.X_test, self.y_test),
                                        batch_size=batch_size, epochs=epochs,
                                        use_multiprocessing=use_multiprocessing,
                                        verbose=verbose, callbacks = [early_stop])

        except Exception as e:
            raise ValueError(f"An error occurred in model compilation or fitting: {e}")

        # Save losses, best loss, and last loss. 
        self.losses = pd.DataFrame(model_loss.history)
        self.best_loss_ = early_stop.best
        self.last_loss_ = self.losses.iloc[-1]
        # Create and store a prediction off X_test.
        self._full_predict(verbose=verbose)

    def _epoch_batch(self):
        """Internal method. returns epoch and batch size based on train data size.

        Returns:
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
        """
        batch_sizes = [2 ** i for i in range(4, 14)] # batch sizes to choose from.
        sqrt_n_features = np.sqrt(len(self.X_columns_))
        min_batch_size = 2 ** int(sqrt_n_features)
        max_batch_size = min_batch_size * sqrt_n_features

        batch_size = next((size for size in batch_sizes if size >= max_batch_size), batch_sizes[-1])
        epochs = min(int(sqrt_n_features) * batch_size, 1000)

        return epochs, batch_size

    def predict(self, X_input:np.ndarray):
        """Predict labels for input data.

        Parameters:
        - X_input (np.ndarray): Input data for prediction.

        Returns:
        - np.ndarray: Predicted labels.
        """
        # Validate X_input.
        if not val.validate_array_length(X_input, len(self.X_columns_)):
            raise ValueError("Error in X input, either not array or unmatched shape.")

        # Individual prediction, scaled or unscaled.
        # If unscaled, check if larger than 1, since there's only minmaxscaler.
        if X_input.max() > 1:
            if len(X_input.shape) == 1:
                X_input = self.scaler.transform(X_input.reshape(1,-1))
            else: # multiple arrarys/rows
                X_input = self.scaler.transform(X_input)
        pred = self.model.predict(X_input)
        return self._clean_pred(pred)

    def _full_predict(self, verbose=1):
        """Internal method. Make predictions on the test data.

        Parameters:
        - verbose (int): Verbosity mode (0, 1, or 2).
        """
        pred = self.model.predict(self.X_test, verbose=verbose)
        self.y_pred = self._clean_pred(pred)

    def _clean_pred(self, pred):
        """Internal method. Clean and format predicted labels.

        Parameters:
        - pred: Predicted labels.

        Returns:
        - np.ndarray: Cleaned and formatted predicted labels.
        """
        if self.model.layers[-1].activation == tf.keras.activations.softmax:
            return np.argmax(pred, axis=-1)
        elif self.model.layers[-1].activation == tf.keras.activations.sigmoid:
            return (pred > 0.5).astype(int)
        else: # continuous
            return pred

    def plot_metric(self, metric:str='loss', figsize:tuple=(5,5)):
        """Plot measured training and validation metrics.

        Parameters:
        - metric (str): Metric within losses to plot.
        - figsize (tuple): Figure size (width, height) in inches.
        """
        try:
            self.losses[[metric, 'val_'+metric]].plot(figsize=figsize)
        except Exception as e:
            raise ValueError(f"An error occurred while plotting the metric, {metric}: {e}")

        plt.xlabel('epochs')

    # Classification
    def classification_metrics(self):
        """Print classification metrics."""
        num_classes = 2 if len(self.y_test.shape) == 1 else self.y_test.shape[1]
        if num_classes > 2:
            y_test = np.argmax(self.y_test, axis=1)
        else: # Binary.
            y_test = self.y_test
        print(classification_report(y_test, self.y_pred))

    # Regression
    def regression_metrics(self):
        """Print regression metrics."""
        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = mean_squared_error(self.y_test, self.y_pred)**0.5
        r2 = r2_score(self.y_test, self.y_pred)
        print(f"MAE: {mae}\nRMSE: {rmse}\nR2 Score: {r2}\n")

    def plot_pred_scatter(self, target_columns: list, title: str):
        """Plot scatter plots of predicted vs. actual values.

        Parameters:
        - target_columns (list): List of target column names.
        - title (str): Title of the plot.
        """
        # Validation.
        # Check if each feature column is a valid column name in self.X_columns_.
        invalid_columns = [col for col in target_columns if col not in self.X_columns_]
        if invalid_columns:
            raise ValueError(f"Invalid feature column(s): {', '.join(invalid_columns)}. "
                            "These columns do not exist in the dataset.")
        # Check title.
        if not isinstance(title, str):
            raise ValueError(f"Title should be str, got '{title}'")

        # Plotting.
        num_subplots = len(target_columns)
        num_rows = math.ceil(num_subplots / 3)  # Adjust 3 based on the desired maximum number of columns
        num_cols = min(num_subplots, 3)
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

        # Flatten the axes if there is only one row or one column
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, feature_column in enumerate(target_columns):
            # Find the index of the feature column in self.X_columns_.
            column_index = self.X_columns_.get_loc(feature_column)
            # Real values.
            ax = axes[i // num_cols, i % num_cols]  # Calculate the subplot index
            ax.plot(self.X_test[:, column_index], self.y_test, 'o', label="Actual", color="blue")
            # Predicted values.
            ax.plot(self.X_test[:, column_index], self.y_pred, 'o', label="Predicted", color="red")
            ax.set_title(f"{feature_column.title()} Relation to {self.y_title}")
            ax.legend()

        # Overall title.
        # Set a common ylabel title for all subplots.
        axes[0, 0].set_ylabel(self.y_title, labelpad=20)
        plt.suptitle(title)
        plt.tight_layout()  # Adjust subplot parameters to give specified padding
        plt.show()

    def plot_residual(self, figsize:tuple=(5,5)):
        """Plots residual errors.

        Parameters:
        - figsize (tuple): Figure size (width, height) in inches.
        """
        # Validate figsize.
        try:
            plt.figure(figsize=figsize)
        except Exception as e:
            raise ValueError(f"An error occurred (figsize): {e}")

        # Calculate residuals in correct dimensions then plot.
        residuals = self.y_test - self.y_pred.flatten()
        plt.scatter(self.y_pred.flatten(), residuals)
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Error Plot')
        plt.show()

    def evaluate_model(self, figsize=(5,5)):
        """Evaluate the model's performance by printing relevant metrics and plots in full.

        Parameters:
        - figsize (tuple): Each figure size (width, height) in inches.
        """
        # y_pred is defined in fit method.
        # All the relevant evaluation methods are called here.
        # Validation of figsize is done within the class methods.
        self.plot_metric(figsize=figsize)

        if self.model.layers[-1].activation == tf.keras.activations.relu:
            # Get and print continuous values/plots.
            self.regression_metrics()
            self.plot_residual(figsize=figsize)
            self.plot_pred_scatter(list(self.X_columns_), "All Features With True and Predicted 'y'")

        else: # Classification.
            # Get and print binary values/plots.
            self.classification_metrics()
            self.plot_metric('accuracy', figsize)

    def save_model(self, filename:str):
        """Save the trained model and scaler to disk.
        Model is saved as '.h5' filetype, scaler is saved seperately as '.pkl' filetype.
        Every other relevant attribute to plotting or predicting is also saved to a '.pkl' filetype.

        Parameters:
        - filename (str): Name of the file to save, not including file extension type.
        """
        # Check if the filename contains any file extension
        if re.search(r'\.[^.]+$', filename):
            raise ValueError("Error: Filename should not contain any file extension.")
        try:
            # Serialize additional attributes
            additional_attributes = {
                'out_activation_': self.out_activation_,
                'layers_': self.layers_,
                'activation_': self.activation_,
                'X_columns_': self.X_columns_,
                'y_classes_': self.y_classes_,
                'best_loss_': self.best_loss_,
                'last_loss_': self.last_loss_,
                'y_title': self.y_title,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'y_pred': self.y_pred,
                'losses': self.losses
            }
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(additional_attributes, f)

            # Save the model
            self.model.save(filename + '.h5')

            # Save the scaler separately so model and scaler COULD be used outside class.
            with open(filename + '_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            print("Saved successfully.")
        except Exception as e: # Catch file, directory, or other errors.
            raise  ValueError(f"An error occurred (save model): {e}")

    @staticmethod
    def load_model(filename:str):
        """Load a saved model, scaler, and attributes from disk.

        Parameters:
        - filename (str): Name of the file to load, not including file extension type.

        Returns:
        - Keras_Model: Loaded Keras_Model object.
        """
        if re.search(r'\.[^.]+$', filename):
            raise ValueError("Error: Filename should not contain any file extension.")
        try:
            # Load additional attributes
            with open(filename + '.pkl', 'rb') as f:
                additional_attributes = pickle.load(f)

            # Load the scaler
            with open(filename + '_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Create a new instance of the class
            obj = Keras_Model(additional_attributes['out_activation_'],
                                additional_attributes['layers_'],
                                additional_attributes['activation_'])

            # Assign deserialized attributes to the new instance
            obj.X_columns_ = additional_attributes['X_columns_']
            obj.y_classes_ = additional_attributes['y_classes_']
            obj.best_loss_ = additional_attributes['best_loss_']
            obj.last_loss_ = additional_attributes['last_loss_']
            obj.y_title = additional_attributes['y_title']
            obj.X_test = additional_attributes['X_test']
            obj.y_test = additional_attributes['y_test']
            obj.y_pred = additional_attributes['y_pred']
            obj.losses = additional_attributes['losses']
            obj.model = load_model(filename + '.h5')
            obj.scaler = scaler

            print("Loaded successfully.")
            return obj
        except Exception as e:
            raise ValueError(f"An error occurred (load model): {e}")
