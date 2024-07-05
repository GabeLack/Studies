import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from validation import Validation as val

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import (LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier

#! future additions?
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score,
                             classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, ConfusionMatrixDisplay)

class MLClass:
    """
    MLClass - A class for handling regression and classification tasks using various machine learning algorithms.
    """
    REGRESSIONS = ['LiR','Lasso','Ridge','ElasticNet','SVR']
    CLASSIFICATIONS = ['LoR','KNN','SVC']
    max_iter = 5_000

    def __init__(self, regression_ml:bool, df:pd.DataFrame, target_column:str, test_size=0.25):
        """
        Initialize MLClass object.

        Parameters:
        - regression_ml (bool): If True, regression task; if False, classification task.
        - df (pd.DataFrame): Input DataFrame containing features and target variable.
        - target_column (str): Name of the target column in the DataFrame.
        - test_size (float, optional): Size of the test set in the train-test split (default is 0.2).
        """
        # Validation for inputs
        # if regression_ml = True, it's regression, otherwise classification
        if not isinstance(regression_ml,bool):
            raise ValueError("Invalid obj, not of type bool.")
        self.regression_ml = regression_ml

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid obj, not of type pd.DataFrame")
        self.df = df

        if target_column not in self.df.columns:
            raise ValueError("Invalid target column, name not in dataframe.")
        self.y = df[target_column]
        self.X = df.drop(target_column, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                        test_size=test_size, random_state=101)

        # Checks if y-label categorical or continuous
        self.multi_class = None
        threshold = 10
        if self.regression_ml:
            # Check if target column is continuous
            if not np.issubdtype(self.y.dtype, np.number) and len(self.y.unique()) < threshold:
                raise ValueError("Target column must be continuous for regression.")
        else:
            # Check if target column is categorical
            if np.issubdtype(self.y.dtype, float):
                raise ValueError("Target column must be categorical for classification")
            # Just used whatever number felt reasonable here
            elif len(self.y.unique()) > threshold:
                raise ValueError(f"Target column is continuous or has more classes than {threshold}.")
            else:
                # Determine if the target variable is binary or multi-class categorical
                self.multi_class = len(self.y.unique()) > 2

        # Dataframes of metrics
        self.metrics = pd.DataFrame()

    def check_missing(self):
        # Checks for missing values, if any, raises userwarning
        missing_values = self.df.isna().sum()
        for column, count in missing_values.items():
            if count > 0:
                raise UserWarning(f"Missing values found in column: {column}")

    def check_feature_types(self):
        # Check for object dtype columns in self.X
        object_columns = self.X.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            raise UserWarning(f"Object dtype columns found in {', '.join(object_columns)}")


#?-----------------------------METHODS FOR REGRESSION----------------------------

    def regression_lir(self, degrees=np.arange(1, 11)):
        """
        Perform linear regression.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default = np.arange(1, 11)).

        Returns:
        - pd.DataFrame: Regression metrics for Linear Regression model.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), LinearRegression())

        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees}

        # Input and then fit model
        self.lir_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                           cv=10, scoring="neg_mean_squared_error")
        self.lir_best_model.fit(self.X_train, self.y_train)

        # Get y_pred and metrics for best found model
        self.lir_y_pred = self.regression_predict_and_metrics(self.lir_best_model, 'LiR')

        return self.metrics[self.metrics['type']=='LiR']

    def regression_ridge(self, degrees=np.arange(1, 11), alphas=np.logspace(-1,1,10)):
        """
        Perform Ridge regression.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - alphas (array-like, optional): Alpha values for Ridge regression (default is np.logspace(-1, 1, 10)).

        Returns:
        - pd.DataFrame: Regression metrics for Ridge Regression model.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints_floats(alphas):
            raise ValueError("Alphas should be a positive list or array of integers or floats.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), Ridge(max_iter=self.max_iter))
        
        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees, 'ridge__alpha': alphas}

        # Input and then fit model
        self.ridge_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                             cv=10, scoring="neg_mean_squared_error")
        self.ridge_best_model.fit(self.X_train, self.y_train)

        # Get y_pred and metrics for best found model
        self.ridge_y_pred = self.regression_predict_and_metrics(self.ridge_best_model, 'Ridge')

        return self.metrics[self.metrics['type']=='Ridge']

    def regression_lasso(self, degrees=np.arange(1, 11), alphas=np.logspace(-1, 1, 10)):
        """
        Perform Lasso regression.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - alphas (array-like, optional): Alpha values for Lasso regression (default is np.logspace(-1, 1, 10)).

        Returns:
        - pd.DataFrame: Regression metrics for Lasso Regression model.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints_floats(alphas):
            raise ValueError("Alphas should be a positive list or array of integers or floats.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), Lasso(max_iter=self.max_iter))

        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees, 'lasso__alpha': alphas}

        # Input and then fit model
        self.lasso_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                             cv=10, scoring="neg_mean_squared_error")
        self.lasso_best_model.fit(self.X_train, self.y_train)

        # Get y_pred and metrics for best found model
        self.lasso_y_pred = self.regression_predict_and_metrics(self.lasso_best_model, 'Lasso')

        return self.metrics[self.metrics['type']=='Lasso']

    def regression_elasticnet(self, degrees=np.arange(1, 11), alphas=np.logspace(-1, 1, 10),
                              l1_ratios=[.1, .5, .7, .9, .95, .99]):
        """
        Perform ElasticNet regression.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - alphas (array-like, optional): Alpha values for ElasticNet regression (default is np.logspace(-1, 1, 10)).
        - l1_ratios (array-like, optional): L1 ratios for ElasticNet regression (default is [.1, .5, .7, .9, .95, .99, 1]).

        Returns:
        - pd.DataFrame: Regression metrics for ElasticNet Regression model.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of positive integers.")
        if not val.validate_pos_ints_floats(alphas):
            raise ValueError("Alphas should be a positive list or array of positive integers or floats.")
        if not (val.validate_pos_ints_floats(l1_ratios) and all(0 < value < 1 for value in l1_ratios)):
            raise ValueError("l1_ratios should be a positive list or array of positive integers or floats between 0 and 1.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), ElasticNet(max_iter=self.max_iter))

        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees,
                      'elasticnet__alpha': alphas, 'elasticnet__l1_ratio': l1_ratios}

        # Input and then fit model
        self.elasticnet_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                                  cv=10, scoring="neg_mean_squared_error")
        self.elasticnet_best_model.fit(self.X_train, self.y_train)

        # Get y_pred and metrics for best found model
        self.elasticnet_y_pred = self.regression_predict_and_metrics(self.elasticnet_best_model, 'ElasticNet')
        
        return self.metrics[self.metrics['type']=='ElasticNet']

    def regression_svr(self, degrees=np.arange(1, 11), c=np.logspace(0, 1, 10),
                       epsilon = [0, 0.001, 0.01, 0.1, 0.5, 1, 2]):
        """
        Perform Support Vector Regression (SVR).

        Parameters:
        - degrees (array-like, optional): Degrees for SVR (default is np.arange(1, 11)).
        - c (array-like, optional): Regularization parameter C for SVR (default is np.logspace(0, 1, 10)).
        - epsilon (array-like, optional): Epsilon values for SVR (default is [0, 0.001, 0.01, 0.1, 0.5, 1, 2]).

        Returns:
        - pd.DataFrame: Regression metrics for SVR model.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints_floats(c):
            raise ValueError("C should be a positive list or array of integers or floats.")
        if not val.validate_pos_ints_floats_w_0(epsilon):
            raise ValueError("Epsilon should be a positive list or array of integers or floats.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(StandardScaler(), SVR()) # degrees included parameter in SVR

        # Define the grid parameters for the hyperparameter search
        grid_param = {'svr__degree': degrees, 'svr__C': c, 'svr__epsilon': epsilon,
                      'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'svr__gamma': ['scale', 'auto']}

        # Input and then fit model
        self.svr_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                           cv=10, scoring="neg_mean_squared_error")
        self.svr_best_model.fit(self.X_train, self.y_train)

        # Get y_pred and metrics for best found model
        self.svr_y_pred = self.regression_predict_and_metrics(self.svr_best_model, 'SVR')

        return self.metrics[self.metrics['type']=='SVR']

    def regression_predict_and_metrics(self, model, algorithm_name):
        """
        Make predictions using the specified regression model and calculate metrics.

        Parameters:
        - model: The trained regression model.
        - algorithm_name (str): The name of the regression algorithm.

        Returns:
        - np.ndarray: Predicted values.
        """
        if algorithm_name not in self.REGRESSIONS:
            raise ValueError(f"{algorithm_name} model not in supported regressions.")
        try:
            y_pred = model.predict(self.X_test)
        except Exception as e:
            raise ValueError(f"Error predicting with the model: {e}")

        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        # Save metrics to the metrics DataFrame
        data = {'type': algorithm_name, 'refit time': model.refit_time_, 'mae': mae, 'rmse': rmse, 'r2': r2}
        # Check if metrics is empty, avoids FutureWarning
        if self.metrics.empty:
            self.metrics = pd.DataFrame([data])
        else:
            self.metrics = pd.concat([self.metrics, pd.DataFrame([data])], ignore_index=True)

        # Return y_pred
        return y_pred

    def regression_plot_resid(self, models=REGRESSIONS):
        """
        Plot residual errors for multiple regression models as subplots.

        Parameters:
        - models (list, optional): List of regression model names to include in the plot
                                (default is ['LiR', 'Lasso', 'Ridge', 'ElasticNet', 'SVR']).

        Raises:
        - ValueError: If a specified model name is not found.
        """
        for model_name in models:
            if model_name not in self.REGRESSIONS:
                raise ValueError(f"Model with name '{model_name}' not found.")

        # Initialize an empty DataFrame to store all y_preds
        all_y_preds = pd.DataFrame(index=self.y_test.index)

        # Collect y_preds for each model
        for model_name in models:
            y_pred = getattr(self, f"{model_name.lower()}_y_pred", None)
            if y_pred is not None:
                all_y_preds[model_name] = y_pred
            else:
                raise ValueError(f"y_pred for {model_name} not found")

        # Create a 2 by 3 subplot grid for the plots
        fig, axes = plt.subplots(3, 2 , figsize=(15, 10))

        # Flatten the axes for easier indexing
        axes = axes.flatten()

        # Plotting with Seaborn regplot for each model
        for i, model_name in enumerate(models):
            # Calculate residual errors for the specific model
            residual_errors = self.y_test - all_y_preds[model_name]

            # Plotting with Seaborn regplot
            sns.scatterplot(x=self.y_test, y=residual_errors, markers=True, ax=axes[i])
            axes[i].axhline(y=0, color="red", linestyle="--")

            # Adding labels and title
            axes[i].set_title(f"Residual Error Plot - {model_name}")
            axes[i].set_xlabel("Actual Values")
            axes[i].set_ylabel("Residual Error")

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Return the figure
        return fig

#?-----------------------------METHODS FOR CLASSIFICATION----------------------------

    def classification_lor(self, degrees=np.arange(1, 6), c=np.logspace(0, 2, 5)):
                           #l1_ratios=[0, .1, .5, .7, .9, .95, .99, 1]):
        """
        Train a logistic regression classifier and perform hyperparameter tuning using GridSearchCV.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - c (array-like, optional): Regularization parameter values for logistic regression (default is np.logspace(0, 4, 10)).

        Returns:
        - str: A report summarizing the logistic regression model's performance.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints_floats(c):
            raise ValueError("C should be a positive list or array of integers or floats.")
        #if not (val.validate_pos_ints_floats_w_0(l1_ratios) and all(value <= 1 for value in l1_ratios)):
        #    raise ValueError("l1_ratios should be a positive list or array of positive integers or floats between 0 and 1.")


        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(),
                             LogisticRegression(max_iter=self.max_iter))

        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees,
                      'logisticregression__C': c,
                      'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
                      'logisticregression__solver': ['lbfgs', 'liblinear',\
                            'newton-cg', 'newton-cholesky', 'saga'],
                      'logisticregression__class_weight': [None, 'balanced']}

        # Input and then fit model
        self.lor_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                           cv=10, scoring="accuracy")
        self.lor_best_model.fit(self.X_train, self.y_train)

        # Get metrics for best found model
        self.lor_y_pred,report = self.classification_predict_and_metrics(self.lor_best_model, 'LoR')

        return f"Logistic regression report:\n{report}"

    def classification_knn(self, degrees=np.arange(1, 11), n_neighbors=np.arange(1, 31)):
        """
        Train a k-nearest neighbors classifier and perform hyperparameter tuning using GridSearchCV.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - n_neighbors (array-like, optional): Number of neighbors to consider (default is np.arange(1, 31)).

        Returns:
        - str: A report summarizing the k-nearest neighbors model's performance.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints(n_neighbors):
            raise ValueError("n_neigbhours should be a positive list or array of integers.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), KNeighborsClassifier())

        # Define the grid parameters for the hyperparameter search
        grid_param = {'polynomialfeatures__degree': degrees,
                      'kneighborsclassifier__n_neighbors': n_neighbors,
                      'kneighborsclassifier__weights': ['uniform','distance'],
                      'kneighborsclassifier__metric':['manhattan', 'euclidean', 'minkowski', 
                                                      'chebyshev', 'mahalanobis', 'seuclidean'],}

        # Input and then fit model
        self.knn_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                           cv=10, scoring="accuracy")
        self.knn_best_model.fit(self.X_train, self.y_train)

        # Get metrics for best found model
        self.knn_y_pred, report = self.classification_predict_and_metrics(self.knn_best_model, 'KNN')

        return f"K nearest neigbhors report:\n{report}"

    def classification_svc(self, degrees=np.arange(1, 11), c=np.logspace(0, 1, 10)):
        """
        Train a support vector classifier and perform hyperparameter tuning using GridSearchCV.

        Parameters:
        - degrees (array-like, optional): Degrees for polynomial features (default is np.arange(1, 11)).
        - c (array-like, optional): Regularization parameter values for support vector classifier (default is np.logspace(0, 1, 10)).

        Returns:
        - str: A report summarizing the support vector classifier's performance.
        """
        # Validation
        if not val.validate_pos_ints(degrees):
            raise ValueError("Degrees should be a positive list or array of integers.")
        if not val.validate_pos_ints_floats(c):
            raise ValueError("C should be a positive list or array of integers or floats.")

        # Create the pipeline using make_pipeline
        pipe = make_pipeline(StandardScaler(), SVC())

        # Define the grid parameters for the hyperparameter search
        grid_param = {'svc__degree': degrees, 'svc__C': c,
                      'svc__kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                      'svc__gamma':['scale', 'auto'],
                      'svc__class_weight':[None, 'balanced']}

        # Input and then fit model
        self.svc_best_model = GridSearchCV(pipe, grid_param, n_jobs=4,
                                           cv=10, scoring="accuracy")
        self.svc_best_model.fit(self.X_train, self.y_train)

        # Get metrics for best found model
        self.svc_y_pred, report = self.classification_predict_and_metrics(self.svc_best_model, 'SVC')

        return f"SVC report:\n{report}"

    def classification_plot_conf(self, models=CLASSIFICATIONS):
        """
        Plot confusion matrices for multiple classification models.

        Parameters:
        - models (list, optional): List of classification model names to include in the plot
                                  (default is ['LoR', 'KNN', 'SVC']).

        Raises:
        - ValueError: If a specified model name is not found.
        """
        # Create a subplot figure with a single column
        fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 4 * len(models)))

        for idx, model_name in enumerate(models):
            if model_name not in self.CLASSIFICATIONS:
                raise ValueError(f"Model with name '{model_name}' not found.")

            # Access the best_estimator_ attribute for the specific model
            model = getattr(self, f"{model_name.lower()}_best_model", None)

            # Get predictions
            y_pred = model.predict(self.X_test)

            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            # Display the confusion matrix with title and labels
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
            
            # Select the subplot for the current model
            ax = axes[idx]

            # Plot the confusion matrix on the selected subplot
            disp.plot(cmap='Blues', values_format='d', ax=ax)  # Adjust cmap and values_format as needed
            ax.set_title(f"Confusion Matrix - {model_name}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        # Adjust layout to prevent overlapping titles
        plt.tight_layout()

        # Return the subplot figure
        return fig

    def classification_predict_and_metrics(self, model, algorithm_name):
        """
        Make predictions and calculate classification metrics for a given classification model.

        Parameters:
        - model: Trained classification model with a predict method.
        - algorithm_name (str): Name of the classification algorithm.

        Returns:
        - Tuple: A tuple containing predicted labels and a classification report.
        """
        if algorithm_name not in self.CLASSIFICATIONS:
            raise ValueError(f"{algorithm_name} model not in supported classifications.")
        try:
            y_pred = model.predict(self.X_test)
        except Exception as e:
            raise ValueError(f"Error predicting with the model: {e}")

        # Calculate metrics
        report = classification_report(self.y_test, y_pred)
        if self.multi_class:
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')
            f1_score_ = f1_score(self.y_test, y_pred, average='macro')
        else:
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1_score_ = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        # Save metrics to the regression_metrics DataFrame
        data = {'type':algorithm_name, 'refit time':model.refit_time_, 'precision':precision,
                'recall':recall, 'f1 score':f1_score_, 'accuracy':accuracy}
        # Check if metrics is empty, avoids FutureWarning
        if self.metrics.empty:
            self.metrics = pd.DataFrame([data])
        else:
            self.metrics = pd.concat([self.metrics, pd.DataFrame([data])], ignore_index=True)

        # Return y_pred
        return y_pred, report

#?-----------------------------METHODS FOR REGRESSION AND CLASSIFICATION----------------------------

    def corr_w_test_score_plot(self, model_name):
        """
        Plot the correlation between hyperparameter columns and mean_test_score for a specific model.

        Parameters:
        - model_name (str): Name of the regression or classification model.

        Raises:
        - ValueError: If the specified model name is not found.
        """
        if model_name not in [*self.REGRESSIONS,*self.CLASSIFICATIONS]:
            raise ValueError(f"Model with name '{model_name}' not found.")
        model = getattr(self, f"{model_name.lower()}_best_model", None)
        
        # Get stats of all estimators, dummify
        estimators = pd.DataFrame(model.cv_results_)
        estimators.drop('params', axis=1, inplace=True)
        estimators = pd.get_dummies(estimators, dtype='int')
        
        # Corr with mean_test_score column
        correlation_with_mean_test_score = estimators.corrwith(estimators['mean_test_score'])
        param_correlations = correlation_with_mean_test_score.filter(like='param_')
        
        # Visualize the correlation matrix using a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=param_correlations, y=param_correlations.index)
        plt.title(f'{model_name} Correlation with Mean Test Score for Parameter Columns')
        plt.xlabel('Correlation')
        plt.ylabel('param_ Columns')
        plt.show()

    def get_best_type(self):
        """
        Get the best-performing algorithm type based on metrics.

        Returns:
        - DataFrame: A DataFrame containing the best-performing algorithm type.
        """
        if self.regression_ml:
            return self.metrics.sort_values('r2', ascending=False).head(1)
        else:  # Classification algorithms
            # Check if balanced
            class_counts = self.y.value_counts()
            is_balanced = (class_counts.min() / class_counts.max()) > 0.5

            if is_balanced:
                return self.metrics.sort_values('accuracy', ascending=False).head(1)
            else:
                # if unbalanced, return algorithm type based on f1 score
                return self.metrics.sort_values('f1 score', ascending=False).head(1)

    def get_best_model(self, model_name):
        """
        Retrieve the best estimator for a specific model from the hyperparameter tuning results.

        Parameters:
        - model_name (str): Name of the regression or classification model.

        Returns:
        - object: The best estimator for the specified model based on hyperparameter tuning.

        Raises:
        - ValueError: If the specified model name is not found in the available models.
        """
        if model_name not in [*self.REGRESSIONS, *self.CLASSIFICATIONS]:
            raise ValueError(f"Model with name '{model_name}' not found.")

        model = getattr(self, f"{model_name.lower()}_best_model", None)

        return model.best_estimator_

    def get_best_params(self, model_name):
        """
        Get the best parameters for a specific model.

        Parameters:
        - model_name (str): Name of the regression or classification model.

        Raises:
        - ValueError: If the specified model name is not found.
        """
        if model_name not in [*self.REGRESSIONS, *self.CLASSIFICATIONS]:
            raise ValueError(f"Model with name '{model_name}' not found.")

        model = getattr(self, f"{model_name.lower()}_best_model", None)

        if model is not None and hasattr(model, 'best_params_'):
            return model.best_params_
        else:
            return None
