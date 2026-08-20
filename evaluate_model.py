import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def evaluate_model_performance(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster.keras'), 
                             data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
                             test_size=168):  # One week of hourly data
    try:
        # Configure proper encoding for console output
        import sys
        import locale
        sys.stdout.reconfigure(encoding='utf-8')
        
        # Load the model
        custom_objects = {
            'mse': mse,
            'loss': mse
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")

        # Load data
        df = pd.read_csv(data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Select demand columns and create time features
        demand_cols = [col for col in df.columns if 'Demand' in col]
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        
        features = demand_cols + ['Hour', 'DayOfWeek']
        
        # Prepare test data
        test_data = df[features].iloc[-test_size:]
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        # Prepare sequences for testing
        sequence_length = 24
        X_test, y_test = [], []
        
        for i in range(len(test_data) - sequence_length):
            X_test.append(scaled_data[-(test_size-i+sequence_length):-(test_size-i)])
            y_test.append(scaled_data[-(test_size-i), :len(demand_cols)])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each band
        metrics = {}
        for i, col in enumerate(demand_cols):
            metrics[col] = {
                'MAE': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            }
        
        # Calculate spectrum efficiency metrics
        def calculate_spectrum_efficiency(actual, predicted):
            # Calculate overallocation (when predicted > actual)
            overallocation = np.maximum(predicted - actual, 0).mean()
            
            # Calculate underallocation (when actual > predicted)
            underallocation = np.maximum(actual - predicted, 0).mean()
            
            # Calculate allocation accuracy
            accuracy = 1 - np.abs(predicted - actual).mean()
            
            return {
                'overallocation': overallocation,
                'underallocation': underallocation,
                'accuracy': accuracy
            }
        
        # Calculate efficiency metrics for each band
        efficiency_metrics = {}
        for i, col in enumerate(demand_cols):
            efficiency_metrics[col] = calculate_spectrum_efficiency(
                y_test[:, i], y_pred[:, i]
            )
            
        # Calculate potential spectrum savings
        total_allocation = y_test.sum()
        optimal_allocation = np.minimum(y_test, y_pred).sum()
        spectrum_savings = ((total_allocation - optimal_allocation) / total_allocation) * 100
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot actual vs predicted for each band
        for i, col in enumerate(demand_cols):
            plt.subplot(3, 2, i+1)
            plt.plot(y_test[:168, i], label='Actual', alpha=0.7)
            plt.plot(y_pred[:168, i], label='Predicted', alpha=0.7)
            plt.title(f'{col} - Actual vs Predicted')
            plt.xlabel('Hours')
            plt.ylabel('Normalized Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'model_evaluation.png'))  # Removed encoding parameter
        plt.close()
        
        results_path = os.path.join(os.path.dirname(__file__), 'evaluation_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("Model Performance Metrics:\n")
            f.write("========================\n")
            for band in metrics:
                f.write(f"\n{band}:\n")
                f.write(f"Mean Absolute Error: {metrics[band]['MAE']:.4f}\n")
                f.write(f"Root Mean Squared Error: {metrics[band]['RMSE']:.4f}\n")
            
            f.write("\nSpectrum Efficiency Metrics:\n")
            f.write("=========================\n")
            for band in efficiency_metrics:
                f.write(f"\n{band}:\n")
                f.write(f"Overallocation: {efficiency_metrics[band]['overallocation']:.4f}\n")
                f.write(f"Underallocation: {efficiency_metrics[band]['underallocation']:.4f}\n")
                f.write(f"Allocation Accuracy: {efficiency_metrics[band]['accuracy']:.4f}\n")
            
            f.write(f"\nPotential Spectrum Savings: {spectrum_savings:.2f}%\n")
        
        # Print results to console
        with open(results_path, 'r', encoding='utf-8') as f:
            print(f.read())
   
        return metrics, efficiency_metrics
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None, None

if __name__ == "__main__":
    metrics, efficiency = evaluate_model_performance()