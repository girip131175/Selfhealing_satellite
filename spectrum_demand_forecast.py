import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def load_and_preprocess_data(data_folder, num_datasets=4):
    """Load and combine multiple spectrum demand datasets"""
    all_data = []
    
    for i in range(1, num_datasets + 1):
        file_path = os.path.join(data_folder, f'spectrum_demand_{i}.csv')
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])
        all_data.append(df)
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Time')
    
    return combined_df

def prepare_features(df):
    # Existing features
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    
    # Additional time-based features
    df['Month'] = df['Time'].dt.month
    df['DayOfMonth'] = df['Time'].dt.day
    df['WeekOfYear'] = df['Time'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Time of day categories
    df['IsPeakHour'] = df['Hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
    df['IsNightTime'] = df['Hour'].isin(range(23, 6)).astype(int)
    
    # Lag features
    for col in [c for c in df.columns if 'Demand' in c]:
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_24'] = df[col].shift(24)
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(6).mean()
    
    # Drop rows with NaN values from lag features
    df.dropna(inplace=True)
    
    return df

def prepare_sequences(data, sequence_length, target_cols):
    """
    Create sequences of data for LSTM training
    
    Args:
        data: DataFrame containing features
        sequence_length: Number of time steps in each sequence
        target_cols: List of column names to predict
        
    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Input sequence
        X.append(data.iloc[i:i+sequence_length].values)
        
        # Target values (next time step for target columns)
        target_values = data.iloc[i+sequence_length][target_cols].values
        y.append(target_values)
    
    return np.array(X), np.array(y)

def create_lstm_model(input_shape, output_shape):
    model = Sequential([
        # Bidirectional LSTM layers for better sequence learning
        Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(512, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(256, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        
        Dense(output_shape, activation='sigmoid')
    ])
    
    # Enhanced optimizer configuration
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    
    return model

def train_demand_forecaster():
    try:
        # Move encoding configuration to the top
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        # Parameters
        sequence_length = 24  # Use 24 hours of data to predict next hour
        train_split = 0.8
        
        # Load and preprocess data
        data_folder = os.path.join(os.path.dirname(__file__), "spectrum_demand_data")
        df = load_and_preprocess_data(data_folder)
        
        # Select demand columns
        demand_cols = [col for col in df.columns if 'Demand' in col]
        
        # Create feature set (you can add more features as needed)
        features = demand_cols + ['Hour', 'DayOfWeek']
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        
        # Prepare sequences
        X, y = prepare_sequences(scaled_df, sequence_length, demand_cols)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train model with Input layer
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        
        # Define model using functional API
        inputs = Input(shape=(sequence_length, len(features)))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(len(demand_cols), activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        # Enhanced training parameters
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-6
        )
        
        model_checkpoint = ModelCheckpoint(
            os.path.join(os.path.dirname(__file__), 'best_model.keras'),  # Changed from .h5 to .keras
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Train with more epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=300,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            shuffle=True
        )

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_test_inv = scaler.inverse_transform(np.concatenate([y_test, np.zeros((len(y_test), 2))], axis=1))[:, :len(demand_cols)]
        y_pred_inv = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((len(y_pred), 2))], axis=1))[:, :len(demand_cols)]
        
        # Save plots with proper encoding
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(demand_cols):
            plt.subplot(3, 2, i+1)
            plt.plot(y_test_inv[:168, i], label='Actual')
            plt.plot(y_pred_inv[:168, i], label='Predicted')
            plt.title(f'{col} - First Week Predictions')
            plt.xlabel('Hours')
            plt.ylabel('Demand')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'demand_predictions.png'))  # Removed encoding parameter
        plt.close()
        
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Plot actual vs predicted for a sample
        plt.figure(figsize=(15, 10))

        # Calculate number of rows needed based on number of demand columns
        num_cols = len(demand_cols)
        num_rows = (num_cols + 1) // 2  # Ceiling division to ensure enough rows

        # Plot actual vs predicted for each band
        for i, col in enumerate(demand_cols):
            plt.subplot(num_rows, 2, i+1)
            plt.plot(y_test[:168, i], label='Actual', alpha=0.7)
            plt.plot(y_pred[:168, i], label='Predicted', alpha=0.7)
            plt.title(f'{col} - Actual vs Predicted (LSTM)')
            plt.xlabel('Hours')
            plt.ylabel('Normalized Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'prediction_results_lstm.png'))
        plt.close()        

        # Save the model with error handling
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'spectrum_demand_forecaster.keras')  # Changed from .h5 to .keras
            model.save(model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as model_error:
            print(f"Error saving model: {str(model_error)}")
        
        # Create forecast function
        def forecast_demand(current_data, horizon=24):
            """Forecast demand for the next 'horizon' hours"""
            predictions = []
            input_seq = current_data[-sequence_length:]
            
            for _ in range(horizon):
                next_pred = model.predict(input_seq.reshape(1, sequence_length, len(features)))
                predictions.append(next_pred[0])
                input_seq = np.roll(input_seq, -1, axis=0)
                input_seq[-1] = np.concatenate([next_pred[0], input_seq[-2, -2:]])
            
            return np.array(predictions)
        
        return model, scaler, forecast_demand
    
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    model, scaler, forecast_demand = train_demand_forecaster()
    print("Model training completed. You can now use the forecast_demand function for predictions.")