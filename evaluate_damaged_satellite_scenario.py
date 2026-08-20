import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import huber
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Create output directories if they don't exist
PLOTS_DIR = os.path.join('satellite_damage_evaluation_plots')
RESULTS_DIR = os.path.join('satellite_damage_evaluation_results')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_spectrum_scenario(
    spectrum_availability,
    model_path=os.path.join('CNN_spectrum_allocation', 'spectrum_demand_forecaster_cnn.keras'),
    data_path=os.path.join('spectrum_demand_data', 'spectrum_demand_5.csv'),
    test_size=168  # One week of hourly data
):
    """
    Evaluate model performance under reduced spectrum availability scenario
    
    Args:
        spectrum_availability: Float between 0 and 1 representing available spectrum percentage
        model_path: Path to the trained CNN model
        data_path: Path to the test data
        test_size: Number of hours to test
        
    Returns:
        Tuple of (performance metrics, efficiency metrics)
    """
    try:
        # Configure proper encoding for Windows
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        
        # Ensure output encoding is set correctly
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')

        # Load model
        custom_objects = {'loss': huber}
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Evaluating scenario with {spectrum_availability*100:.0f}% spectrum availability")

        # Load and prepare data
        df = pd.read_csv(data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Get original demand columns
        demand_cols = [col for col in df.columns if 'Demand' in col]
        
        # Store original demand values for comparison
        original_demands = {}
        for col in demand_cols:
            original_demands[col] = df[col].copy()
        
        # Adjust spectrum capacity to simulate satellite damage
        for col in demand_cols:
            df[col] = df[col] * spectrum_availability
        
        # Create time features
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        df['Month'] = df['Time'].dt.month
        df['DayOfMonth'] = df['Time'].dt.day
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsPeakHour'] = df['Hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        df['IsNightTime'] = df['Hour'].isin(range(23, 6)).astype(int)

        # Create lag features
        for col in demand_cols:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_2'] = df[col].shift(2)
            df[f'{col}_lag_3'] = df[col].shift(3)
            df[f'{col}_lag_24'] = df[col].shift(24)
            df[f'{col}_rolling_mean_3h'] = df[col].rolling(3).mean()
            df[f'{col}_rolling_mean_6h'] = df[col].rolling(6).mean()
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_3'] = df[col].diff(3)

        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != 'Time']
        test_data = df[feature_cols].iloc[-test_size:]
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        
        # Prepare sequences
        sequence_length = 48
        X_test, y_test = [], []
        
        for i in range(len(test_data) - sequence_length):
            X_test.append(scaled_data[-(test_size-i+sequence_length):-(test_size-i)])
            y_test.append(scaled_data[-(test_size-i), :len(demand_cols)])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        for i, col in enumerate(demand_cols):
            metrics[col] = {
                'MAE': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])),
                'R2': r2_score(y_test[:, i], y_pred[:, i])
            }
        
        # Calculate spectrum efficiency
        efficiency_metrics = {}
        for i, col in enumerate(demand_cols):
            efficiency_metrics[col] = calculate_spectrum_efficiency(
                y_test[:, i], y_pred[:, i], spectrum_availability
            )
        
        # Save results
        save_evaluation_results(metrics, efficiency_metrics, spectrum_availability)
        
        # Plot results
        plot_predictions(y_test, y_pred, demand_cols, spectrum_availability)
                
        return metrics, efficiency_metrics
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def calculate_spectrum_efficiency(actual, predicted, spectrum_availability):
    """Calculate spectrum efficiency metrics considering reduced availability"""
    # Adjust maximum possible allocation
    max_allocation = spectrum_availability
    
    # Calculate overallocation (when predicted > actual or > max_allocation)
    overallocation = np.maximum(
        np.minimum(predicted, max_allocation) - actual, 
        0
    ).mean()
    
    # Calculate underallocation (when actual > predicted)
    underallocation = np.maximum(actual - predicted, 0).mean()
    
    # Calculate allocation accuracy considering the capacity limit
    accuracy = 1 - np.mean(
        np.abs(
            np.minimum(predicted, max_allocation) - actual
        )
    )
    
    # Calculate spectrum utilization
    utilization = np.mean(predicted) / max_allocation
    
    # Calculate efficiency (how well we use the limited spectrum)
    efficiency = np.mean(np.minimum(predicted, actual)) / np.mean(actual) if np.mean(actual) > 0 else 0
    
    return {
        'overallocation': overallocation,
        'underallocation': underallocation,
        'accuracy': accuracy,
        'utilization': utilization,
        'efficiency': efficiency
    }

def save_evaluation_results(metrics, efficiency_metrics, spectrum_availability):
    """Save evaluation results to a file"""
    output_path = os.path.join(RESULTS_DIR, f'spectrum_evaluation_{int(spectrum_availability*100)}percent.txt')
    
    # Open file with UTF-8 encoding explicitly
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results (Spectrum Availability: {spectrum_availability*100:.0f}%)\n")
        f.write("="*50 + "\n\n")
        
        # Write performance metrics
        f.write("Performance Metrics:\n")
        f.write("-"*20 + "\n")
        for band, metric in metrics.items():
            f.write(f"\n{band}:\n")
            for name, value in metric.items():
                f.write(f"{name}: {value:.4f}\n")
        
        # Write efficiency metrics
        f.write("\nSpectrum Efficiency Metrics:\n")
        f.write("-"*20 + "\n")
        for band, metric in efficiency_metrics.items():
            f.write(f"\n{band}:\n")
            for name, value in metric.items():
                f.write(f"{name}: {value:.4f}\n")

def plot_predictions(y_test, y_pred, demand_cols, spectrum_availability):
    """Plot actual vs predicted values with spectrum limitation"""
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(demand_cols):
        plt.subplot(3, 2, i+1)
        plt.plot(y_test[:168, i], label='Actual', alpha=0.7)
        plt.plot(y_pred[:168, i], label='Predicted', alpha=0.7)
        plt.axhline(y=spectrum_availability, color='r', linestyle='--', 
                   label='Spectrum Limit')
        plt.title(f'{col} - Actual vs Predicted')
        plt.xlabel('Hours')
        plt.ylabel('Normalized Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'predictions_{int(spectrum_availability*100)}percent.png'))
    plt.close()

def plot_comparative_metrics(all_metrics, all_efficiency_metrics, availabilities):
    """Plot comparative metrics across different spectrum availability scenarios"""
    # Plot performance metrics
    plt.figure(figsize=(15, 10))
    
    # Get all demand bands
    demand_bands = list(all_metrics[availabilities[0]].keys())
    
    # Plot MAE for each band
    for i, band in enumerate(demand_bands):
        plt.subplot(3, 2, i+1)
        
        mae_values = [all_metrics[avail][band]['MAE'] for avail in availabilities]
        rmse_values = [all_metrics[avail][band]['RMSE'] for avail in availabilities]
        
        x_labels = [f"{int(avail*100)}%" for avail in availabilities]
        x_pos = np.arange(len(availabilities))
        
        width = 0.35
        plt.bar(x_pos - width/2, mae_values, width, label='MAE')
        plt.bar(x_pos + width/2, rmse_values, width, label='RMSE')
        
        plt.title(f'{band} - Error Metrics')
        plt.xlabel('Spectrum Availability')
        plt.ylabel('Error Value')
        plt.xticks(x_pos, x_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comparative_performance_metrics.png'))
    plt.close()
    
    # Plot efficiency metrics
    plt.figure(figsize=(15, 15))
    
    metrics_to_plot = ['accuracy', 'utilization', 'efficiency', 'overallocation', 'underallocation']
    
    for i, metric_name in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1)
        
        for band in demand_bands:
            values = [all_efficiency_metrics[avail][band][metric_name] for avail in availabilities]
            plt.plot(x_labels, values, marker='o', label=band)
        
        plt.title(f'Comparative {metric_name.capitalize()}')
        plt.xlabel('Spectrum Availability')
        plt.ylabel(metric_name.capitalize())
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comparative_efficiency_metrics.png'))
    plt.close()

def run_all_scenarios():
    """Run evaluation for all spectrum availability scenarios"""
    # Define spectrum availability scenarios
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    all_metrics = {}
    all_efficiency_metrics = {}
    
    for availability in availabilities:
        print(f"\n{'='*50}")
        print(f"Evaluating {availability*100:.0f}% spectrum availability scenario")
        print(f"{'='*50}\n")
        
        metrics, efficiency_metrics = evaluate_spectrum_scenario(availability)
        
        all_metrics[availability] = metrics
        all_efficiency_metrics[availability] = efficiency_metrics
    
    # Create comparative plots
    plot_comparative_metrics(all_metrics, all_efficiency_metrics, availabilities)
    
    print("\nAll scenarios evaluated successfully!")
    print(f"Results saved in {RESULTS_DIR}")
    print(f"Plots saved in {PLOTS_DIR}")
    
    return all_metrics, all_efficiency_metrics

if __name__ == "__main__":
    try:
        all_metrics, all_efficiency_metrics = run_all_scenarios()
        print("Evaluation completed. Check the output files for results.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)