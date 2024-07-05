import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

"""
Ok so about all this. I'm not happy with how this turned out, I would like to take even more time developing it, 
and maybe even integrating the singleRNN methods into the KerasANN from exam 1. Then have multiRNN call on that class
appropiately.
There were a few more methods I wanted to add to these two classes as well, for saving/loading for ease of testing.
As well as more convenient metric plotting and test vs pred plotting, since as I've shown as well as I could in the
'test.ipynb' doc, it doesnt FEEL convenient enough. It also just looks kinda ugly. I've put the unfinished methods into
the 'removed methods.ipynb' doc.
Lastly, didn't really have time to double check or stress test all the inputs into the methods, so it's probably kinda
weak in that regard.
"""

class SingleRNN:
    def __init__(self, layers: tuple,
                 length,
                 n_features: int = 1,
                 activations: str = 'tanh',
                 out_activation: str = 'relu') -> None:
        """Initialize the SingleRNN model.

        Parameters:
        - layers (tuple): Defines the architecture of the network. 
                          Positive int = LSTM layer,
                          Negative int = Dense layer,
                          0 to 1 float = Dropout layer.
        - length (int): Length of the input sequences.
        - n_features (int): Number of features in the input data. Default is 1.
        - activations (str): Activation function for the hidden Dense layers. Default is 'tanh'.
        - out_activation (str): Activation function for the output layer. Default is 'relu'.
        """
        self.layers = layers
        self.length = length
        self.activations = activations
        self.out_activation = out_activation
        self.n_features = n_features

        # Validate layers and find the last LSTM layer index.
        self._validate_layers(layers)
        last_lstm_index = self._last_positive_index(layers)
        
        # Initialize the Sequential model.
        model = Sequential()
        
        # Create the model layers based on the specified configuration.
        for i, neurons in enumerate(layers):
            if isinstance(neurons, int) and neurons > 0:  # LSTM layer.
                if i == 0 and i == last_lstm_index:  # Single LSTM layer.
                    model.add(LSTM(neurons, return_sequences=False, input_shape=(length, n_features)))
                elif i == 0:  # First LSTM layer in a multi-layer setup.
                    model.add(LSTM(neurons, return_sequences=True, input_shape=(length, n_features)))
                elif i == last_lstm_index:  # Last LSTM layer.
                    model.add(LSTM(neurons, return_sequences=False))
                else:  # Middle LSTM layers.
                    model.add(LSTM(neurons, return_sequences=True))
            elif isinstance(neurons, int) and neurons < 0:  # Dense layer.
                if i == len(layers) - 1:  # Last Dense layer with output activation.
                    model.add(Dense(-neurons, activation=out_activation))
                else:  # Hidden Dense layers.
                    model.add(Dense(-neurons, activation=activations))
            elif isinstance(neurons, float) and 1 > neurons > 0:  # Dropout layer.
                model.add(Dropout(neurons))
            else:
                raise ValueError(f"SingleRNN: error in layer creation, invalid: {neurons}.")
        
        self.model = model

    def _last_positive_index(self, layers):
        """Find the index of the last positive integer in the layers tuple.

        Parameters:
        - layers (tuple): The layers configuration.

        Returns:
        - int: Index of the last positive integer.
        """
        return max(idx for idx, value in enumerate(layers) if isinstance(value, int) and value > 0)

    def _validate_layers(self, layers):
        """Validate the layers configuration.

        Parameters:
        - layers (tuple): The layers configuration.

        Raises:
        - ValueError: If layers is empty or contains invalid values.
        """
        if not layers or not isinstance(layers, (tuple, list)):
            raise ValueError("Layers must be a non-empty tuple or list")
        for layer in layers:
            if not (isinstance(layer, int) or (isinstance(layer, float) and 0 < layer < 1)):
                raise ValueError(f"Invalid layer configuration: {layer}")

    def fit(self, train: pd.DataFrame, test: pd.DataFrame, 
            loss=Huber(), batch_size: int = 1, epochs: int = 25, optimizer: str = 'adam',
            monitor: str = 'val_loss', patience=2, mode: str = 'auto',
            metrics: list = None, verbose: int = 1):
        """Fit the model on the training data.

        Parameters:
        - train (pd.DataFrame): Training data.
        - test (pd.DataFrame): Testing data.
        - loss: Loss function. Default is Huber().
        - batch_size (int): Batch size. Default is 1.
        - epochs (int): Number of epochs. Default is 25.
        - optimizer (str): Optimizer. Default is 'adam'.
        - monitor (str): Metric to monitor for early stopping. Default is 'val_loss'.
        - patience (int): Number of epochs with no improvement after which training will be stopped. Default is 2.
        - mode (str): One of {'auto', 'min', 'max'}. Default is 'auto'.
        - metrics (list): List of metrics to be evaluated by the model during training and testing.
        - verbose (int): Verbosity mode. Default is 1.
        
        Raises:
        - ValueError: If there are missing values in the data or the data is not scaled.
        """
        if self._validate_if_isna(train, test):
            raise ValueError("An error occurred: data contains missing values.")
        if not self._validate_if_scaling(train, test):
            raise ValueError("An error occurred: data should not be scaled.")
        self.column_names = train.columns

        if not isinstance(batch_size, int):
            raise ValueError("An error occurred: batch_size is not int.")
        self.batch_size = batch_size

        # Scale and store data.
        self.scaler = MinMaxScaler()
        self.train = train
        self.test = test
        self.scaled_train = self.scaler.fit_transform(train)
        self.scaled_test = self.scaler.transform(test)
        
        # Make train and validation generators.
        self.train_generator = TimeseriesGenerator(self.scaled_train, self.scaled_train, length=self.length, batch_size=self.batch_size)
        self.val_generator = TimeseriesGenerator(self.scaled_test, self.scaled_test, length=self.length, batch_size=self.batch_size)

        # try compile, earlystopping, and then fitting. Raise valueerror if it fails.
        try:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            early_stop = EarlyStopping(monitor=monitor, mode=mode,
                                       patience=patience, verbose=verbose,
                                       restore_best_weights=True)

            self.model_loss = self.model.fit(self.train_generator, validation_data=self.val_generator,
                                             epochs=epochs, verbose=verbose, callbacks=[early_stop])
            
            self.losses = pd.DataFrame(self.model_loss.history)
        except Exception as e:
            raise ValueError(f"An error occurred: model compilation or fitting: {e}")

    def _validate_if_isna(self, train, test):
        """Check if there are any missing values in the train or test data.

        Parameters:
        - train (pd.DataFrame): Training data.
        - test (pd.DataFrame): Testing data.

        Returns:
        - bool: True if there are missing values, False otherwise.
        """
        return train.isna().sum().sum() > 0 or test.isna().sum().sum() > 0

    def _validate_if_scaling(self, train, test):
        """Validate if the data is scaled properly.

        Parameters:
        - train (pd.DataFrame): Training data.
        - test (pd.DataFrame): Testing data.

        Returns:
        - bool: True if the data is not scaled, False otherwise.
        """
        return train.abs().max().max() > 1 or test.abs().max().max() > 1  # Since only minmax scaling, cannot be larger than 1.

    def predict(self, predict_length: int = 'test_length'):
        """Generate predictions using the trained model.

        Parameters:
        - predict_length (int): Number of steps to predict. Default is 'test_length'.

        Returns:
        - pd.DataFrame: DataFrame containing the predictions.
        
        Raises:
        - ValueError: If predict_length is invalid.
        """
        if predict_length == 'test_length':
            predict_length = len(self.scaled_test)
        if not (isinstance(predict_length, int) and predict_length > 0):
            raise ValueError(f"An error occurred: invalid predict length '{predict_length}'")

        test_predictions = list()
        first_eval_batch = self.scaled_train[-self.length:]
        current_batch = first_eval_batch.reshape((1, self.length, self.n_features))
        
        for i in range(predict_length):
            current_pred = self.model.predict(current_batch)[0]  # Get prediction 1 time stamp ahead.
            test_predictions.append(current_pred)  # Store prediction.
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)  # Update batch to include pred and drop first val.

        preds = self.scaler.inverse_transform(test_predictions)
        return pd.DataFrame(preds, columns=self.column_names)

    @staticmethod
    def _validate_columns(columns, 
                          column_names):
        """Validate that the specified columns exist in the column names.

        Parameters:
        - columns (list): List of columns to validate.
        - column_names (list): List of column names to check against.

        Raises:
        - ValueError: If any column is not found in column names.
        """
        invalid_columns = set(columns) - set(column_names)
        if invalid_columns:
            raise ValueError(f"Invalid column(s) not found in column names: {', '.join(invalid_columns)}")

    def plot_metric(self, metric: str = 'loss', 
                    figsize: tuple = (5, 5)):
        """Plot the training and validation metrics.

        Parameters:
        - metric (str): Metric within losses to plot. Default is 'loss'.
        - figsize (tuple): Figure size (width, height) in inches. Default is (5, 5).

        Raises:
        - ValueError: If an error occurs while plotting the metric.
        """
        try:
            self.losses[[metric, 'val_'+metric]].plot(figsize=figsize)
        except Exception as e:
            raise ValueError(f"An error occurred while plotting the metric, {metric}: {e}")

        plt.xlabel('epochs')

    def plot_prediction_w_test(self, predictions: pd.DataFrame, 
                               columns: list = 'auto',
                               figsize: tuple = (5, 5), 
                               title: str = "Plot predicted vs test"):
        """Plot the test data and predictions for the specified columns.

        Parameters:
        - predictions (pd.DataFrame): DataFrame containing the predictions.
        - columns (list): List of columns to plot. If 'auto', plot all columns. Default is 'auto'.
        - figsize (tuple): Figure size (width, height) in inches. Default is (5, 5).
        - title (str): Title of the plot. Default is "Plot predicted vs test".

        Raises:
        - ValueError: If test and predictions are not the same length.
        """
        if columns == 'auto':
            columns = self.column_names
        SingleRNN._validate_columns(columns, self.column_names)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in columns:
            # Copy current column.
            test_w_pred = self.test[column].copy()
            
            # Make sure the test and predictions are the same length.
            if len(test_w_pred) != len(predictions[column]):
                raise ValueError("An error occurred: test and predictions must be of the same length.")
            
            # Combine test and predictions into a single DataFrame.
            combined_df = pd.DataFrame({'Test': test_w_pred, 'Predictions': predictions[column].values}, index=test_w_pred.index)
            
            # Plot the combined DataFrame.
            combined_df.plot(ax=ax)
        
        # Set the title of the plot.
        ax.set_title(title)
        plt.show()

class MultiRNN:
    def __init__(self, layers: np.ndarray,
                 length: int,
                 n_features: int,
                 activations: np.ndarray = 'tanh',  # Default to 'tanh' per layer per feature.
                 out_activation: str = 'relu'):
        """Initialize the MultiRNN model.

        Parameters:
        - layers (np.ndarray): Defines the architecture of the network. 
                               Send in a tuple containing positive for LSTM, negative for hidden, 
                               and 0 < x < 1 for dropout.
        - length (int): Length of the input sequences.
        - n_features (int): Number of features in the input data.
        - activations (np.ndarray): Activation functions for the hidden Dense layers. Default is 'tanh'.
        - out_activation (str): Activation function for the output layer. Default is 'relu'.
        """
        self.all_layers = layers
        self.length = length
        self.n_features = n_features
        self.activations = activations
        self.out_activation = out_activation

        self._check_last_element()

        self.multi_models = []
        for feature in range(self.n_features):
            if any(isinstance(item, tuple) for item in self.all_layers):
                model = SingleRNN(self.all_layers[feature], length=self.length,
                                  activations=activations, out_activation=out_activation)
                self.multi_models.append(model)
            else:
                model = SingleRNN(self.all_layers, length=self.length,
                                  activations=activations, out_activation=out_activation)
                self.multi_models.append(model)

    def _check_last_element(self):
        """Check if the last element in each tuple within layers is -1.
        This multiRNN requires each model to end with an output layer of 1 neuron.

        Raises:
        - ValueError: If the last element is not -1.
        """
        if any(isinstance(item, tuple) for item in self.all_layers):
            for layer_tuple in self.all_layers:
                if layer_tuple[-1] != -1:
                    raise ValueError("Last element in each tuple within layers must be -1")
        else:
            if self.all_layers[-1] != -1:
                raise ValueError("Last element in layers must be -1")

    def fit(self, train: pd.DataFrame, test: pd.DataFrame,
            loss=Huber(), batch_size=1, epochs=25, optimizer='adam',
            monitor='val_loss', patience=2, mode='auto',
            metrics=None, verbose=1):
        """Fit the model on the training data.

        Parameters:
        - train (pd.DataFrame): Training data.
        - test (pd.DataFrame): Testing data.
        - loss: Loss function. Default is Huber().
        - batch_size (int): Batch size. Default is 1.
        - epochs (int): Number of epochs. Default is 25.
        - optimizer (str): Optimizer. Default is 'adam'.
        - monitor (str): Metric to monitor for early stopping. Default is 'val_loss'.
        - patience (int): Number of epochs with no improvement after which training will be stopped. Default is 2.
        - mode (str): One of {'auto', 'min', 'max'}. Default is 'auto'.
        - metrics (list): List of metrics to be evaluated by the model during training and testing.
        - verbose (int): Verbosity mode. Default is 1.

        Raises:
        - ValueError: If an error occurs during fitting.
        """
        self.train_dfs, self.test_dfs = self._split_data_per_column(train, test)
        self.column_names = train.columns

        # Check iterable length, duplicate if singular.
        loss = self._check_and_duplicate_list(loss)
        batch_size = self._check_and_duplicate_list(batch_size)
        epochs = self._check_and_duplicate_list(epochs)
        optimizer = self._check_and_duplicate_list(optimizer)

        # A specific loss, batch_size, epochs, and optimizer can be used per model.
        for i, (model, loss_val, batch_size_val, epochs_val, optimizer_val) in enumerate(zip(self.multi_models, loss, batch_size, epochs, optimizer)):
            try:
                model.fit(train=self.train_dfs[i], test=self.test_dfs[i], loss=loss_val,
                          batch_size=batch_size_val, epochs=epochs_val, optimizer=optimizer_val,
                          monitor=monitor, patience=patience, mode=mode,
                          metrics=metrics, verbose=verbose)
            except KeyError as e:
                raise ValueError(f"An error occurred: fitting the model '{model}' due to a KeyError with the key: {e}")
            except Exception as e:
                raise ValueError(f"An error occurred: fitting the model '{model}': {e}")

    def _split_data_per_column(self, train, test):
        """Split the data into separate dataframes per column.

        Parameters:
        - train (pd.DataFrame): Training data.
        - test (pd.DataFrame): Testing data.

        Returns:
        - train_dfs (list): List of dataframes for training data.
        - test_dfs (list): List of dataframes for testing data.
        """
        train_dfs = []
        test_dfs = []
        for i in range(train.shape[1]):  # iterate over columns.
            train_dfs.append(train.iloc[:, [i]])
            test_dfs.append(test.iloc[:, [i]])
        return train_dfs, test_dfs  # Returns lists of dataframes.

    def _check_and_duplicate_list(self, obj):
        """Check if the input object is iterable and duplicate if singular.

        Parameters:
        - obj: Object to check.

        Returns:
        - list: List of duplicated objects or original list if iterable.

        Raises:
        - ValueError: If there is a bad length in iterable.
        """
        try:
            # Just raise TypeError if it's just a string so it gets caught in except block below.
            if isinstance(obj, str):
                raise TypeError
            # if it's not iterable, it won't have a length, and it'll raise a TypeError.
            if len(obj) != self.n_features:
                # if it can iterate, and it's length is wrong, raise an error
                raise ValueError("An error occurred: bad length in iterable")
        except TypeError:
            return [obj] * self.n_features
        else:
            return obj

    def predict(self, predict_length: int = 'test_length'):
        """Generate predictions for the test data.

        Parameters:
        - predict_length (int): Length of the prediction sequence. Default is 'test_length', which predicts the length of the test data.

        Returns:
        - preds (pd.DataFrame): DataFrame containing the predictions for each model.

        Raises:
        - ValueError: If an error occurs during prediction.
        """
        preds = pd.DataFrame()
        for model in self.multi_models:
            try:
                model_preds = model.predict(predict_length)
                preds = pd.concat([preds, model_preds], axis=1)  # Concatenate along columns axis
            except Exception as e:
                raise ValueError(f"An error occurred while predicting with model '{model}': {e}")
        return preds

    # To get metrics or predictions vs tests of each model in multi_models, use the singleRNN methods.
    # As shown in the test.ipynb doc.