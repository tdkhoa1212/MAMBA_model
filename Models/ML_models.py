from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import matplotlib.pyplot as plt
from Utils.Loading_data import Get_data
from Utils.Evaluation_metrics import information_coefficient, rank_information_coefficient

warnings.filterwarnings("ignore")


def run_ML_models():
    # Model performance storage
    dataset_results = {}
    processed_data = Get_data('Data')
    for name, data in processed_data.items():
        # Get features and labels
        X_train, y_train = data['X_train'].reshape(data['X_train'].shape[0], -1), data['y_train']
        X_test, y_test = data['X_test'].reshape(data['X_test'].shape[0], -1), data['y_test']
        
        # Define the models
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'KNN': KNeighborsRegressor()
        }
        
        # Store the results for this dataset
        model_results = {}

        # Create a figure with subplots
        num_models = len(models)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()  # Flatten to make indexing easier

        plot_index = 0  # Index for the subplot

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            # Calculate RMSE, IC, and RIC for each model
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            ic = information_coefficient(y_test, pred)
            ric = rank_information_coefficient(y_test, pred)
            
            # Store the metrics for each model
            model_results[model_name] = {
                'RMSE': rmse,
                'IC': ic,
                'RIC': ric
            }
            
            # Plot the model predictions vs true values in a subplot
            axes[plot_index].plot(pred, label='Prediction', color='blue')
            axes[plot_index].plot(y_test, linestyle='--', label='True', color='red')
            axes[plot_index].set_title(f"{model_name} Predictions vs True")
            axes[plot_index].set_xlabel("Index")
            axes[plot_index].set_ylabel("Values")
            axes[plot_index].legend()
            axes[plot_index].grid(True)

            plot_index += 1
        
        # Adjust layout
        plt.suptitle(f'{name}')
        plt.tight_layout()
        plt.show()
        
        # Store the model results for this dataset
        dataset_results[name] = model_results
    
    # Print the results
    for name, results in dataset_results.items():
        print(f"{name} Dataset Performance:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    RMSE: {metrics['RMSE']:.4f}")
            print(f"    IC: {metrics['IC']:.4f}")
            print(f"    RIC: {metrics['RIC']:.4f}")
        print('-' * 50)
