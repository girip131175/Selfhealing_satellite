import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Input, Concatenate, LSTM, Bidirectional, Add, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import huber
import matplotlib.pyplot as plt
import os

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
    """Prepare features for CNN model with enhanced recent data emphasis"""
    # Time-based features
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Month'] = df['Time'].dt.month
    df['DayOfMonth'] = df['Time'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Time of day categories
    df['IsPeakHour'] = df['Hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
    df['IsNightTime'] = df['Hour'].isin(range(23, 6)).astype(int)
    
    # Lag features for demand columns with more emphasis on recent data
    for col in [c for c in df.columns if 'Demand' in c]:
        # Add more recent lags to capture immediate patterns
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_2'] = df[col].shift(2)
        df[f'{col}_lag_3'] = df[col].shift(3)
        df[f'{col}_lag_24'] = df[col].shift(24)
        
        # Add weighted rolling means that emphasize recent values
        df[f'{col}_rolling_mean_3h'] = df[col].rolling(3).mean()
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(6).mean()
        
        # Add rate of change features to capture trends
        df[f'{col}_diff_1'] = df[col].diff(1)
        df[f'{col}_diff_3'] = df[col].diff(3)
    
    # Drop rows with NaN values from lag features
    df.dropna(inplace=True)
    
    return df

def prepare_sequences(data, sequence_length, target_cols):
    """Create sequences of data for CNN training"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Input sequence
        X.append(data.iloc[i:i+sequence_length].values)
        
        # Target values (next time step for target columns)
        target_values = data.iloc[i+sequence_length][target_cols].values
        y.append(target_values)
    
    return np.array(X), np.array(y)

def create_cnn_model(input_shape, output_shape):
    """Create an enhanced CNN model for time series forecasting with deeper architecture and attention mechanism"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First branch: CNN for local patterns
    # Immediate temporal relationships layer
    conv_immediate = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inputs)
    conv_immediate = BatchNormalization()(conv_immediate)
    
    # First block with residual connection
    conv1 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', dilation_rate=1)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    
    # Second block with residual connection
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2_2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2_2 = BatchNormalization()(conv2_2)
    # Add residual connection
    conv2 = Add()([conv2, conv2_2])
    conv2 = Dropout(0.2)(conv2)
    
    # Third block with residual connection
    conv3 = Conv1D(filters=256, kernel_size=4, activation='relu', padding='same', dilation_rate=4)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3_2 = Conv1D(filters=256, kernel_size=4, activation='relu', padding='same', dilation_rate=4)(conv3)
    conv3_2 = BatchNormalization()(conv3_2)
    # Add residual connection
    conv3 = Add()([conv3, conv3_2])
    conv3 = Dropout(0.2)(conv3)
    
    # Fourth block
    conv4 = Conv1D(filters=512, kernel_size=5, activation='relu', padding='same', dilation_rate=8)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.3)(conv4)
    
    # Attention mechanism
    attention = Attention()([conv4, conv4])
    
    # Second branch: LSTM for temporal dependencies
    lstm_branch = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Bidirectional(LSTM(128, return_sequences=False))(lstm_branch)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    
    # Flatten CNN branch
    cnn_flat = Flatten()(attention)
    
    # Concatenate CNN features with immediate features
    cnn_immediate_flat = Flatten()(conv_immediate)
    combined_cnn = Concatenate()([cnn_flat, cnn_immediate_flat])
    
    # First dense layer for CNN branch
    cnn_dense = Dense(512, activation='relu')(combined_cnn)
    cnn_dense = BatchNormalization()(cnn_dense)
    cnn_dense = Dropout(0.4)(cnn_dense)
    
    # Combine CNN and LSTM branches
    combined = Concatenate()([cnn_dense, lstm_branch])
    
    # Deep dense layers with residual connections
    dense1 = Dense(512, activation='relu')(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(256, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    
    dense3 = Dense(128, activation='relu')(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.2)(dense3)
    
    # Output layer
    outputs = Dense(output_shape, activation='linear')(dense3)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with a lower learning rate
    optimizer = Adam(learning_rate=0.0003)  # Adjusted learning rate
    model.compile(
        optimizer=optimizer,
        loss=huber,
        metrics=['mae', 'mse']
    )
    
    return model

def train_demand_forecaster():
    """Train CNN model for spectrum demand forecasting"""
    try:
        # Configure encoding for console output
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
        # Parameters
        sequence_length = 48  # Use 24 hours of data to predict next hour
        train_split = 0.8
        
        # Load and preprocess data
        data_folder = os.path.join(os.path.dirname(__file__), "spectrum_demand_data")
        df = load_and_preprocess_data(data_folder)
        df = prepare_features(df)
        
        # Select demand columns
        demand_cols = [col for col in df.columns if 'Demand' in col and not ('lag' in col or 'rolling' in col)]
        
        # Create feature set
        feature_cols = [col for col in df.columns if col != 'Time']
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)
        
        # Prepare sequences
        X, y = prepare_sequences(scaled_df, sequence_length, demand_cols)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train model
        input_shape = (sequence_length, len(feature_cols))
        output_shape = len(demand_cols)
        
        model = create_cnn_model(input_shape, output_shape)
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1),
            ModelCheckpoint(
                filepath=os.path.join(os.path.dirname(__file__), 'best_cnn_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model with more epochs and smaller batch size
        print("Training CNN model...")
        history = model.fit(
            X_train, y_train,
            epochs=400,  # Increased from 300
            batch_size=16,  # Decreased from 32 for better generalization
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        print("Evaluating model...")
        test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), 'spectrum_demand_forecaster_cnn.keras')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'training_history_cnn.png'))
        
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
            plt.title(f'{col} - Actual vs Predicted (CNN)')
            plt.xlabel('Hours')
            plt.ylabel('Normalized Demand')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'prediction_results_cnn.png'))
        
        return model, scaler, demand_cols
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # Fix console encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Training CNN Spectrum Demand Forecaster ===")
    model, scaler, demand_cols = train_demand_forecaster()
    print("Training complete!")