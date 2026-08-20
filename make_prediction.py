import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import mse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

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

def predict_future_demand(model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster.keras'), 
                         data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
                         horizon=24):
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
        last_time = pd.to_datetime(df['Time'].iloc[-1])
        future_times = pd.date_range(start=last_time, periods=horizon+1, freq='h')[1:]

        # Create DataFrame with predictions
        pred_df = pd.DataFrame(predictions_inv, columns=demand_cols, index=future_times)
        
        # Plot future predictions
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(demand_cols):
            plt.subplot(3, 2, i+1)
            plt.plot(pred_df.index, pred_df[col], marker='o', color='blue', alpha=0.7)
            plt.title(f'{col} - Future Predictions (LSTM)', fontsize=12)
            plt.xlabel('Time', fontsize=10)
            plt.ylabel('Demand (%)', fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        plt.savefig(os.path.join(os.path.dirname(__file__), 'future_predictions_lstm.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save predictions to CSV
        pred_df.to_csv(os.path.join(os.path.dirname(__file__), 'future_predictions.csv'))
        
        print("\nPrediction Summary:")
        print("-------------------")
        for col in demand_cols:
            print(f"\n{col}:")
            print(f"Average predicted demand: {pred_df[col].mean():.2f}%")
            print(f"Peak demand: {pred_df[col].max():.2f}% at {pred_df[col].idxmax()}")
            print(f"Minimum demand: {pred_df[col].min():.2f}% at {pred_df[col].idxmin()}")
        
        return pred_df
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # Make predictions for the next 24 hours
    predictions = predict_future_demand(
        model_path=os.path.join(os.path.dirname(__file__),'spectrum_demand_forecaster.keras'),
        data_path=os.path.join(os.path.dirname(__file__), "spectrum_demand_data", "spectrum_demand_1.csv"),
        horizon=24
    )
    
    if predictions is not None:
        print("\nPredictions saved to 'future_predictions.csv'")
        print("Visualization saved to 'future_predictions.png'")