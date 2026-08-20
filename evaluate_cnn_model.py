'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import huber
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def evaluate_cnn_model(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster_cnn.keras'), 
                      data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
                      test_size=168):  # One week of hourly data
    try:
        # Configure proper encoding for console output
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        
        # Load the model
        custom_objects = {
            'loss': huber
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("CNN model loaded successfully")
        
        print(f"Model summary:")
        model.summary()

        # Load data
        df = pd.read_csv(data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Select demand columns and create time features
        demand_cols = [col for col in df.columns if 'Demand' in col]
        print(f"Found {len(demand_cols)} demand columns: {demand_cols}")
        
        # Create time features
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        df['Month'] = df['Time'].dt.month
        df['DayOfMonth'] = df['Time'].dt.day
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsPeakHour'] = df['Hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        df['IsNightTime'] = df['Hour'].isin(range(23, 6)).astype(int)
        
        # Create enhanced lag features to match training
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
        
        # Select all features except Time
        feature_cols = [col for col in df.columns if col != 'Time']
        
        # Get expected input shape from model
        expected_features = None
        try:
            model_config = model.get_config()
            input_layer_config = model_config['layers'][0]['config']
            if 'batch_input_shape' in input_layer_config:
                expected_features = input_layer_config['batch_input_shape'][2]
            else:
                input_shape = model.input_shape
                if isinstance(input_shape, tuple) and len(input_shape) >= 3:
                    expected_features = input_shape[2]
        except:
            # Fallback to a fixed value
            expected_features = 103
            
        print(f"Model expects {expected_features} features, data has {len(feature_cols)} features")
        
        # Adjust feature count if needed
        if len(feature_cols) != expected_features:
            if len(feature_cols) < expected_features:
                # Add dummy features
                for i in range(len(feature_cols), expected_features):
                    dummy_name = f"dummy_feature_{i}"
                    df[dummy_name] = 0.0
                    feature_cols.append(dummy_name)
            else:
                # Keep only essential features
                essential_cols = demand_cols.copy()
                for col in ['Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'IsWeekend', 'IsPeakHour', 'IsNightTime']:
                    essential_cols.append(col)
                
                # Add lag features until we reach expected count
                for col in demand_cols:
                    for feature in [f'{col}_lag_1', f'{col}_lag_2', f'{col}_lag_3', f'{col}_lag_24', 
                                   f'{col}_rolling_mean_3h', f'{col}_rolling_mean_6h',
                                   f'{col}_diff_1', f'{col}_diff_3']:
                        if len(essential_cols) < expected_features and feature in df.columns:
                            essential_cols.append(feature)
                
                # If we still need more features, add other available features
                remaining_cols = [col for col in feature_cols if col not in essential_cols]
                essential_cols.extend(remaining_cols[:expected_features - len(essential_cols)])
                
                feature_cols = essential_cols[:expected_features]
                
            print(f"Adjusted feature count to {len(feature_cols)}")
        
        # Prepare test data
        test_data = df[feature_cols].iloc[-test_size:]
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        
        # Prepare sequences for testing
        sequence_length = 48
        X_test, y_test = [], []
        
        for i in range(len(test_data) - sequence_length):
            X_test.append(scaled_data[-(test_size-i+sequence_length):-(test_size-i)])
            y_test.append(scaled_data[-(test_size-i), :len(demand_cols)])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Before making predictions
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check if prediction shape matches expected output shape
        if y_pred.shape[1] != len(demand_cols):
            print(f"Warning: Model output shape {y_pred.shape} doesn't match demand columns {len(demand_cols)}")
            
            # If model outputs more values than we have demand columns, use only the first len(demand_cols) outputs
            if y_pred.shape[1] > len(demand_cols):
                print(f"Truncating predictions from {y_pred.shape[1]} to {len(demand_cols)} columns")
                y_pred = y_pred[:, :len(demand_cols)]
            # If model outputs fewer values than demand columns, pad with zeros
            else:
                print(f"Padding predictions from {y_pred.shape[1]} to {len(demand_cols)} columns")
                padded_pred = np.zeros((y_pred.shape[0], len(demand_cols)))
                padded_pred[:, :y_pred.shape[1]] = y_pred
                y_pred = padded_pred
        
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
            plt.title(f'{col} - Actual vs Predicted (CNN)')
            plt.xlabel('Hours')
            plt.ylabel('Normalized Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'cnn_model_evaluation.png'))
        plt.close()
        
        results_path = os.path.join(os.path.dirname(__file__), 'cnn_evaluation_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("CNN Model Performance Metrics:\n")
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
    metrics, efficiency = evaluate_cnn_model()

'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import huber
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def evaluate_cnn_model(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster_cnn.keras'), 
                      data_path=r"spectrum_demand_1.csv",
                      test_size=168):  # One week of hourly data
    try:
        # Configure proper encoding for console output
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        
        # Load the model
        custom_objects = {
            'loss': huber
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("CNN model loaded successfully")
        
        print(f"Model summary:")
        model.summary()

        # Load data
        df = pd.read_csv(data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Select demand columns and create time features
        demand_cols = [col for col in df.columns if 'Demand' in col]
        print(f"Found {len(demand_cols)} demand columns: {demand_cols}")
        
        # Create time features
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        df['Month'] = df['Time'].dt.month
        df['DayOfMonth'] = df['Time'].dt.day
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsPeakHour'] = df['Hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        df['IsNightTime'] = df['Hour'].isin(range(23, 6)).astype(int)
        
        # Create enhanced lag features to match training
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
        
        # Select all features except Time
        feature_cols = [col for col in df.columns if col != 'Time']
        
        # Get expected input shape from model
        expected_features = None
        try:
            model_config = model.get_config()
            input_layer_config = model_config['layers'][0]['config']
            if 'batch_input_shape' in input_layer_config:
                expected_features = input_layer_config['batch_input_shape'][2]
            else:
                input_shape = model.input_shape
                if isinstance(input_shape, tuple) and len(input_shape) >= 3:
                    expected_features = input_shape[2]
        except:
            # Fallback to a fixed value
            expected_features = 103
            
        print(f"Model expects {expected_features} features, data has {len(feature_cols)} features")
        
        # Adjust feature count if needed
        if len(feature_cols) != expected_features:
            if len(feature_cols) < expected_features:
                # Add dummy features
                for i in range(len(feature_cols), expected_features):
                    dummy_name = f"dummy_feature_{i}"
                    df[dummy_name] = 0.0
                    feature_cols.append(dummy_name)
            else:
                # Keep only essential features
                essential_cols = demand_cols.copy()
                for col in ['Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'IsWeekend', 'IsPeakHour', 'IsNightTime']:
                    essential_cols.append(col)
                
                # Add lag features until we reach expected count
                for col in demand_cols:
                    for feature in [f'{col}_lag_1', f'{col}_lag_2', f'{col}_lag_3', f'{col}_lag_24', 
                                   f'{col}_rolling_mean_3h', f'{col}_rolling_mean_6h',
                                   f'{col}_diff_1', f'{col}_diff_3']:
                        if len(essential_cols) < expected_features and feature in df.columns:
                            essential_cols.append(feature)
                
                # If we still need more features, add other available features
                remaining_cols = [col for col in feature_cols if col not in essential_cols]
                essential_cols.extend(remaining_cols[:expected_features - len(essential_cols)])
                
                feature_cols = essential_cols[:expected_features]
                
            print(f"Adjusted feature count to {len(feature_cols)}")
        
        # Prepare test data
        test_data = df[feature_cols].iloc[-test_size:]
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        
        # Prepare sequences for testing
        sequence_length = 48
        X_test, y_test = [], []
        
        for i in range(len(test_data) - sequence_length):
            X_test.append(scaled_data[-(test_size-i+sequence_length):-(test_size-i)])
            y_test.append(scaled_data[-(test_size-i), :len(demand_cols)])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Before making predictions
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Check if prediction shape matches expected output shape
        if y_pred.shape[1] != len(demand_cols):
            print(f"Warning: Model output shape {y_pred.shape} doesn't match demand columns {len(demand_cols)}")
            
            # If model outputs more values than we have demand columns, use only the first len(demand_cols) outputs
            if y_pred.shape[1] > len(demand_cols):
                print(f"Truncating predictions from {y_pred.shape[1]} to {len(demand_cols)} columns")
                y_pred = y_pred[:, :len(demand_cols)]
            # If model outputs fewer values than demand columns, pad with zeros
            else:
                print(f"Padding predictions from {y_pred.shape[1]} to {len(demand_cols)} columns")
                padded_pred = np.zeros((y_pred.shape[0], len(demand_cols)))
                padded_pred[:, :y_pred.shape[1]] = y_pred
                y_pred = padded_pred
        
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
            plt.title(f'{col} - Actual vs Predicted (CNN)')
            plt.xlabel('Hours')
            plt.ylabel('Normalized Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'cnn_model_evaluation.png'))
        plt.close()
        
        results_path = os.path.join(os.path.dirname(__file__), 'cnn_evaluation_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("CNN Model Performance Metrics:\n")
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

    metrics, efficiency = evaluate_cnn_model()
