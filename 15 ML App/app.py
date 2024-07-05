### APP section ###

# Ask if this data will be determined via regression or classification.

# Ask for filepath.

# Show column list, user makes choice of target column.

# Validate if target column is OK with the choice of regression or classification when initialized.
# Raises error if incorrect ML type with target column.
# Raises error and prompts for correction if NaN and/or feature columns are object dtype.

# MLclass init with these inputs, then get results of each model.
# Show metrics of each model.
# Show resid error plots if regression, conf matrix if classification.

# Give recommended model based on r2 score (regression) and accuracy or f1 score (classification)
# Show correlations of mean test score on recommended model with parameters if user wants to know
# which parameters were the most important.

# Show best params of recommended model

# Show plot of corr with mean test score on best advised model type.

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from mlclass import MLClass
import matplotlib.pyplot as plt
import io
from joblib import dump
import time

class MLApp:
    ml_class = None
    plots = []  # To store all plots

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MLApp")

        # Variables to store user inputs
        self.data_type = tk.StringVar()
        self.file_path = tk.StringVar()
        self.target_column = tk.StringVar()

        # Create a Text widget for displaying output
        self.output_text = tk.Text(self.root, height=20, width=60)
        self.output_text.pack()

        # Create a progress bar
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress_bar.pack()

        # UI elements
        self.create_widgets()

    def create_widgets(self):
        # Choose data type (Regression/Classification)
        tk.Label(self.root, text="Choose data type:").pack()
        tk.Radiobutton(self.root, text="Regression", variable=self.data_type, value="regression").pack()
        tk.Radiobutton(self.root, text="Classification", variable=self.data_type, value="classification").pack()

        # Select CSV file
        tk.Label(self.root, text="Select CSV file:").pack()
        tk.Button(self.root, text="Browse", command=self.browse_file).pack()

        # Show column list and choose target column
        tk.Button(self.root, text="Show Column List", command=self.show_column_list).pack()

        # Validate and initialize MLClass
        tk.Button(self.root, text="Initialize MLClass", command=self.initialize_ml_class).pack()

        # Get results and show plots
        tk.Button(self.root, text="Show Results", command=self.show_results).pack()

        # Show column list and choose target column
        tk.Button(self.root, text="Recommended Model", command=self.recommended_model).pack()

        # Show column list and choose target column
        tk.Button(self.root, text="Choose Model", command=self.choose_model).pack()

        # Close button
        tk.Button(self.root, text="Close", command=self.root.destroy).pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.file_path.set(file_path)

    def show_column_list(self):
        # Load CSV file and show column list
        file_path = self.file_path.get()
        if file_path:
            df = pd.read_csv(file_path)
            column_list = list(df.columns)

            # Check if an OptionMenu already exists
            if hasattr(self, 'target_option_menu'):
                tk.messagebox.showwarning("Warning", "Target column selection can only be done once.")
                return

            # Create a new OptionMenu
            target_column = tk.StringVar(value=column_list[0])
            tk.Label(self.root, text="Choose target column:").pack()
            option_menu = tk.OptionMenu(self.root, target_column, *column_list)
            option_menu.pack()

            # Store the OptionMenu reference in the root
            self.target_option_menu = option_menu
            self.target_column = target_column
        else:
            tk.messagebox.showerror("Error", "Please select a CSV file.")

    def initialize_ml_class(self):
        # Validate and initialize MLClass
        if self.data_type.get() and self.file_path.get() and self.target_column.get():
            df = pd.read_csv(self.file_path.get())
            try:
                self.ml_class = MLClass(regression_ml=(self.data_type.get() == "regression"),
                                df=df, target_column=self.target_column.get())

                # Check for missing values
                try:
                    self.ml_class.check_missing()
                except UserWarning as e:
                    # Ask the user about fixing/dropping missing values
                    response = tk.messagebox.askquestion("Missing Values",
                                f"{str(e)}\n\nDo you want to fix/drop missing values?")
                    if response == 'yes':
                        # Reset the index after dropping rows with missing values
                        df.dropna(inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        # Reinitialize with corrected df
                        self.ml_class = MLClass(regression_ml=(self.data_type.get() == "regression"),
                                                df=df,target_column=self.target_column.get())
                    else:
                        # Ask user to choose new file
                        self.browse_file()
                        return  # Stop execution here if the user chooses a new file

                # Check for object dtype columns
                try:
                    self.ml_class.check_feature_types()
                except UserWarning as e:
                    # Ask the user about using get_dummies on object feature columns
                    response = tk.messagebox.askquestion("Object Feature Columns",
                                f"{str(e)}\n\nDo you want to use get_dummies on object feature columns?")
                    if response == 'yes':
                        # run get dummies on columns other than target column
                        df = pd.get_dummies(df, columns=[col for col in df.columns if \
                                                         col != self.target_column.get()],
                                            drop_first=True, dtype='int')
                        self.ml_class = MLClass(regression_ml=(self.data_type.get() == "regression"),
                                    df=df, target_column=self.target_column.get())
                    else:
                        # Ask user to choose new file
                        self.browse_file()
                        return

                tk.messagebox.showinfo("Success", "MLClass initialized successfully.")
            except ValueError as e:
                tk.messagebox.showerror("Error", str(e))
        else:
            tk.messagebox.showerror("Error", "Please provide all required inputs.")

    def show_results(self):
        # Configure the progress bar for determinate mode
        self.progress_bar.config(mode="determinate", maximum=100, value=0)

        # Get results and show plots
        if hasattr(self, 'ml_class'):
            # Capture printed output to display in the app
            printed_output = io.StringIO()

            total_steps = 0

            start_time = time.time()  # Record the start time

            if self.ml_class.regression_ml:
                # Regression methods
                regression_methods = [
                    self.ml_class.regression_lir,
                    self.ml_class.regression_ridge,
                    self.ml_class.regression_lasso,
                    self.ml_class.regression_elasticnet,
                    self.ml_class.regression_svr
                ]
                total_steps += len(regression_methods)

                for method in regression_methods:
                    # Perform regression method
                    method()
                    # Increment progress bar
                    self.progress_bar.step(100 / total_steps)
                    # Update the window
                    self.root.update()

                # Plot and store residual errors for all regression models
                plot = self.ml_class.regression_plot_resid()
            else:
                # Classification methods
                classification_methods = [
                    self.ml_class.classification_lor,
                    self.ml_class.classification_knn,
                    self.ml_class.classification_svc
                ]
                total_steps += len(classification_methods)

                for method in classification_methods:
                    # Perform classification method
                    method()
                    # Increment progress bar
                    self.progress_bar.step(100 / total_steps)
                    # Update the window
                    self.root.update()

                plot = self.ml_class.classification_plot_conf()

            # Print cumulative metric DataFrames
            metrics_str = str(self.ml_class.metrics)
            print(metrics_str, file=printed_output)

            # Get the captured printed content
            printed_output = printed_output.getvalue()

            # Display the figure containing subplots
            plt.show(block=False)  # Show the plot without blocking

            # Display the printed output in the app
            self.output_text.insert(tk.END, printed_output)

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            self.output_text.insert(tk.END, f"\n\nExecution Time: {elapsed_time:.2f} seconds\n")
        else:
            tk.messagebox.showerror("Error", "Please initialize MLClass first.")

        # Stop the progress bar when the process is done
        self.progress_bar.stop()

    def recommended_model(self):
        # After show results, recommend the best-performing model based on metrics
        if hasattr(self, 'ml_class'):
            # Get the recommended model type
            recommended_model_type = self.ml_class.get_best_type()['type'].iloc[0]

            # Display the recommended model type in the text box
            self.output_text.insert(tk.END, f"\nRecommended Best-Performing Model Type: {recommended_model_type}\n")

            # Display the best parameters of the recommended model
            best_params = self.ml_class.get_best_params(recommended_model_type)
            self.output_text.insert(tk.END, f"Best Parameters for {recommended_model_type}:\n{best_params}\n\n")

            # Plot the correlation with mean test score for parameter columns for the recommended model
            self.ml_class.corr_w_test_score_plot(recommended_model_type)
        else:
            tk.messagebox.showerror("Error", "Please initialize MLClass first.")

    def choose_model(self):
        # After show results, keep those boxes and plots visible, let the user pick from a list of model types
        if hasattr(self, 'ml_class'):
            # Get a list of available model types
            model_types = self.ml_class.REGRESSIONS if self.ml_class.regression_ml else self.ml_class.CLASSIFICATIONS

            # Create a variable to store the selected model type
            selected_model_type = tk.StringVar(self.root)
            selected_model_type.set(model_types[0])  # Set the default value

            # Create an OptionMenu to choose the model type
            tk.Label(self.root, text="Choose Model Type:").pack()
            model_type_menu = tk.OptionMenu(self.root, selected_model_type, *model_types)
            model_type_menu.pack()

            # Wait for the user to choose a model type
            self.root.wait_variable(selected_model_type)

            # Get the selected model type
            selected_model_type = selected_model_type.get()

            # Get the best parameters for the selected model type
            best_params = self.ml_class.get_best_params(selected_model_type)

            # Display the best parameters in the text box
            self.output_text.insert(tk.END, f"Best Parameters for {selected_model_type}:\n{best_params}\n\n")

            # Plot the correlation with mean test score for parameter columns
            self.ml_class.corr_w_test_score_plot(selected_model_type)

            # Ask the user for the file path and name
            file_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                    filetypes=[("Joblib files", "*.joblib"),
                                                                ("All files", "*.*")])

            # Dump the best model to the chosen file
            dump(self.ml_class.get_best_model(selected_model_type), file_path)
            
            tk.messagebox.showinfo("Success", f"Model saved successfully to {file_path}.")
        else:
            tk.messagebox.showerror("Error", "Please initialize MLClass first.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MLApp()
    app.run()
