'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import huber
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def load_latest_data(data_path, sequence_length=48):
    """Load and prepare the most recent data for prediction"""
    # Load the data
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Select demand columns and create time features
    demand_cols = [col for col in df.columns if 'Demand' in col]
    
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
    
    # Create feature set
    features = [col for col in df.columns if col != 'Time']
    
    # Get the latest sequence
    latest_data = df[features].iloc[-sequence_length:]
    
    return latest_data, demand_cols, df

def predict_future_demand(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster_cnn.keras'), 
                         data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
                         horizon=24,
                         sequence_length=48):
    try:
        # Load the model with custom objects
        custom_objects = {
            'loss': huber
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("CNN model loaded successfully")
        
        # Get model input shape using a more reliable method
        model_config = model.get_config()
        input_layer_config = model_config['layers'][0]['config']
        expected_features = None
        
        # Try different ways to get the input shape
        if 'batch_input_shape' in input_layer_config:
            expected_features = input_layer_config['batch_input_shape'][2]
        else:
            # Alternative approach: use model.input_shape
            input_shape = model.input_shape
            if isinstance(input_shape, tuple) and len(input_shape) >= 3:
                expected_features = input_shape[2]
            else:
                # Fallback to a fixed value based on your training data
                expected_features = 73  # Use the number from your training data
                print("Could not determine input shape from model, using default value")
        
        print(f"Model expects input with {expected_features} features")
        
        # Load and prepare latest data
        latest_data, demand_cols, df = load_latest_data(data_path, sequence_length)
        
        # Check if feature count matches
        actual_features = latest_data.shape[1]
        print(f"Data has {actual_features} features")
        
        if actual_features != expected_features:
            print(f"WARNING: Feature count mismatch. Model expects {expected_features} features, but data has {actual_features}.")
            print("Attempting to adjust feature set...")
            
            # If we have fewer features than expected, add dummy features
            if actual_features < expected_features:
                for i in range(actual_features, expected_features):
                    feature_name = f"dummy_feature_{i}"
                    latest_data[feature_name] = 0.0
            # If we have more features than expected, drop extra features
            elif actual_features > expected_features:
                # Keep demand columns and drop extra derived features
                essential_cols = demand_cols.copy()
                for col in ['Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'IsWeekend', 'IsPeakHour', 'IsNightTime']:
                    if col in latest_data.columns:
                        essential_cols.append(col)
                
                # Add lag features until we reach the expected count
                for col in demand_cols:
                    for feature in [f'{col}_lag_1', f'{col}_lag_24', f'{col}_rolling_mean_6h']:
                        if len(essential_cols) < expected_features and feature in latest_data.columns:
                            essential_cols.append(feature)
                
                # If we still need more features, add other available features
                remaining_cols = [col for col in latest_data.columns if col not in essential_cols]
                essential_cols.extend(remaining_cols[:expected_features - len(essential_cols)])
                
                # Keep only the essential columns
                latest_data = latest_data[essential_cols]
        
        # Initialize scaler and scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(latest_data)
        
        # Make predictions
        predictions = []
        input_seq = scaled_data
        last_time = pd.to_datetime(df['Time'].iloc[-1])
        
        # Get the actual number of output features from the model
        output_shape = model.output_shape
        num_outputs = output_shape[1] if isinstance(output_shape, tuple) else 6
        print(f"Model produces {num_outputs} outputs")
        
        # Ensure we're only using the number of demand columns that match the model output
        if len(demand_cols) != num_outputs:
            print(f"WARNING: Model outputs {num_outputs} values but we have {len(demand_cols)} demand columns")
            if len(demand_cols) > num_outputs:
                print(f"Using only the first {num_outputs} demand columns")
                demand_cols = demand_cols[:num_outputs]
            else:
                print(f"Model outputs more values than we have demand columns. Using all {len(demand_cols)} demand columns.")
        
        for i in range(horizon):
            # Predict next hour
            next_pred = model.predict(input_seq.reshape(1, len(input_seq), input_seq.shape[1]), verbose=0)
            
            # Ensure prediction matches the number of demand columns
            if next_pred.shape[1] != len(demand_cols):
                print(f"Reshaping prediction from {next_pred.shape[1]} to {len(demand_cols)} outputs")
                # If model outputs more values than demand columns, truncate
                if next_pred.shape[1] > len(demand_cols):
                    next_pred = next_pred[:, :len(demand_cols)]
                # If model outputs fewer values than demand columns, pad with zeros
                else:
                    padded_pred = np.zeros((next_pred.shape[0], len(demand_cols)))
                    padded_pred[:, :next_pred.shape[1]] = next_pred
                    next_pred = padded_pred
            
            predictions.append(next_pred[0])
            
            # Calculate next hour and day
            next_time = last_time + pd.Timedelta(hours=i+1)
            next_hour = next_time.hour
            next_day = next_time.dayofweek
            next_month = next_time.month
            next_day_of_month = next_time.day
            next_is_weekend = 1 if next_day in [5, 6] else 0
            next_is_peak = 1 if next_hour in [9, 10, 11, 14, 15, 16] else 0
            next_is_night = 1 if next_hour in range(23, 6) else 0
            
            # Create new row with predicted values and time features
            new_row = np.zeros(input_seq.shape[1])
            
            # Fill in demand values (only up to the number of outputs from the model)
            for j in range(min(len(demand_cols), len(next_pred[0]))):
                new_row[j] = next_pred[0][j]
            
            # Dynamically determine time feature indices based on column names
            time_features = {
                'Hour': next_hour / 24.0,
                'DayOfWeek': next_day / 7.0,
                'Month': next_month / 12.0,
                'DayOfMonth': next_day_of_month / 31.0,
                'IsWeekend': next_is_weekend,
                'IsPeakHour': next_is_peak,
                'IsNightTime': next_is_night
            }
            
            # Set time features using column names instead of fixed indices
            for feature_name, value in time_features.items():
                if feature_name in latest_data.columns:
                    idx = latest_data.columns.get_loc(feature_name)
                    new_row[idx] = value
            
            # Fill in time features (simplified for demonstration)
            time_feature_indices = {
                'Hour': len(demand_cols),
                'DayOfWeek': len(demand_cols) + 1,
                'Month': len(demand_cols) + 2,
                'DayOfMonth': len(demand_cols) + 3,
                'IsWeekend': len(demand_cols) + 4,
                'IsPeakHour': len(demand_cols) + 5,
                'IsNightTime': len(demand_cols) + 6
            }
            
            # Set time features
            new_row[time_feature_indices['Hour']] = next_hour / 24.0
            new_row[time_feature_indices['DayOfWeek']] = next_day / 7.0
            new_row[time_feature_indices['Month']] = next_month / 12.0
            new_row[time_feature_indices['DayOfMonth']] = next_day_of_month / 31.0
            new_row[time_feature_indices['IsWeekend']] = next_is_weekend
            new_row[time_feature_indices['IsPeakHour']] = next_is_peak
            new_row[time_feature_indices['IsNightTime']] = next_is_night
            
            # Set lag features
            for j, col in enumerate(demand_cols):
                lag_1_idx = latest_data.columns.get_loc(f"{col}_lag_1")
                lag_24_idx = latest_data.columns.get_loc(f"{col}_lag_24")
                rolling_idx = latest_data.columns.get_loc(f"{col}_rolling_mean_6h")
                
                new_row[lag_1_idx] = next_pred[0][j]  # Current prediction becomes lag_1
                
                if i < 23:  # For first 23 predictions, lag_24 comes from original data
                    new_row[lag_24_idx] = input_seq[-23+i, j]
                else:  # After that, use our own predictions
                    new_row[lag_24_idx] = predictions[i-23][j]
                
                # Simple rolling mean calculation
                if i < 5:
                    # Mix of original data and predictions
                    values = list(input_seq[-5+i:, j]) + [p[j] for p in predictions[:i+1]]
                else:
                    # All from predictions
                    values = [p[j] for p in predictions[i-5:i+1]]
                
                new_row[rolling_idx] = np.mean(values)
            
            # Update input sequence
            input_seq = np.roll(input_seq, -1, axis=0)
            input_seq[-1] = new_row
        
        # Convert predictions back to original scale
        predictions = np.array(predictions)
        
        # Create dummy array with all features to use with inverse_transform
        dummy_full_features = np.zeros((len(predictions), input_seq.shape[1]))
        dummy_full_features[:, :len(demand_cols)] = predictions
        
        # Inverse transform
        predictions_inv = scaler.inverse_transform(dummy_full_features)[:, :len(demand_cols)]
        
        # Create timestamp index for predictions
        future_times = pd.date_range(start=last_time, periods=horizon+1, freq='H')[1:]
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(predictions_inv, columns=demand_cols)
        pred_df.insert(0, 'Time', future_times)
        
        # Save predictions to CSV
        output_path = os.path.join(os.path.dirname(__file__), 'cnn_future_predictions.csv')
        pred_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Plot predictions
        plt.figure(figsize=(15, 10))
        
        # Calculate number of rows needed based on number of demand columns
        num_cols = len(demand_cols)
        num_rows = (num_cols + 1) // 2  # Ceiling division to ensure enough rows
        
        for i, col in enumerate(demand_cols):
            plt.subplot(num_rows, 2, i+1)
            plt.plot(pred_df['Time'], pred_df[col], marker='o', linestyle='-')
            plt.title(f'CNN Predicted {col}')
            plt.xlabel('Time')
            plt.ylabel('Demand')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'cnn_demand_forecast.png'))
        plt.close()
        
        return pred_df
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # Fix console encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== CNN-based Spectrum Demand Prediction ===")
    predictions = predict_future_demand()
    
    if predictions is not None:
        print("\nPrediction Summary:")
        print(predictions.describe())
    
    print("\nPrediction complete!")

'''



import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import mse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def load_latest_data(data_path, sequence_length=24):
    """Load and prepare the most recent data for prediction"""
    # Load the data
    df = pd.read_csv(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Select demand columns and create time features
    demand_cols = [col for col in df.columns if 'Demand' in col]
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    
    # Create feature set
    features = demand_cols + ['Hour', 'DayOfWeek']
    
    # Get the latest sequence
    latest_data = df[features].iloc[-sequence_length:]
    
    return latest_data, demand_cols, df  # Return df as well

def predict_future_demand(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster_cnn.keras'), 
                         data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
                         horizon=24,
                         start_datetime=None):
    try:
        # Load the model with custom objects
        custom_objects = {
            'mse': mse,
            'loss': mse,
            'mae': MeanAbsoluteError()
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        
        # Load and prepare latest data
        latest_data, demand_cols, df = load_latest_data(data_path)

        # Initialize scaler and scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(latest_data)
        
        # Make predictions
        predictions = []
        input_seq = scaled_data
        last_hour = latest_data['Hour'].iloc[-1]
        last_day = latest_data['DayOfWeek'].iloc[-1]
        
        for i in range(horizon):
            # Predict next hour
            next_pred = model.predict(input_seq.reshape(1, len(input_seq), input_seq.shape[1]), verbose=0)
            predictions.append(next_pred[0])
            
            # Calculate next hour and day
            next_hour = (last_hour + i + 1) % 24
            next_day = (last_day + (last_hour + i + 1) // 24) % 7
            
            # Update input sequence with proper dimensions
            new_row = np.concatenate([
                next_pred[0],
                np.array([next_hour / 24.0, next_day / 7.0])
            ])
            input_seq = np.roll(input_seq, -1, axis=0)
            input_seq[-1] = new_row
        
        # Convert predictions back to original scale
        predictions = np.array(predictions)
        predictions_full = np.concatenate([
            predictions,
            np.zeros((len(predictions), 2))  # Add dummy values for Hour and DayOfWeek
        ], axis=1)
        predictions_inv = scaler.inverse_transform(predictions_full)[:, :len(demand_cols)]
        
        # Create timestamp index for predictions
        if start_datetime is None:
            last_time = pd.to_datetime(df['Time'].iloc[-1])
        else:
            last_time = start_datetime - timedelta(hours=1)
            
        future_times = pd.date_range(start=last_time, periods=horizon+1, freq='h')[1:]

        # Create DataFrame with predictions
        pred_df = pd.DataFrame(predictions_inv, columns=demand_cols)
        pred_df['Time'] = future_times
        
        # Format the time column for better display in Streamlit
        pred_df['Time'] = pred_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
        
        return pred_df
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Make predictions for the next 24 hours
    predictions = predict_future_demand(
        model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster_cnn.keras'),
        data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
        horizon=24
    )
    
    if predictions is not None:
        print("\nPredictions generated successfully")
        
        # Plot the predictions
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(predictions.columns):
            if 'Time' not in col:
                plt.subplot(3, 2, i+1)
                plt.plot(predictions['Time'], predictions[col], marker='o', color='blue', alpha=0.7)
                plt.title(f'{col} - Future Predictions (CNN)', fontsize=12)
                plt.xlabel('Time', fontsize=10)
                plt.ylabel('Demand (%)', fontsize=10)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
        
        plt.savefig(os.path.join(os.path.dirname(__file__), 'future_predictions_cnn.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()