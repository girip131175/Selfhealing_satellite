import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from pathlib import Path

class SpectrumDemandPredictor:
    def __init__(self, input_window=24, prediction_horizon=6, latent_dim=100, batch_size=64):
        """
        Initialize the GAN-based spectrum demand predictor
        
        Args:
            input_window: Number of time steps to use as input (e.g., 24 hours of history)
            prediction_horizon: Number of time steps to predict into the future
            latent_dim: Dimension of the latent space for the generator
            batch_size: Batch size for training
        """
        self.input_window = input_window
        self.prediction_horizon = prediction_horizon
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_features = None  # Will be determined from the data
        
        # Placeholder for models
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.predictive_model = None
        
        # Placeholder for scalers
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        
    def _build_generator(self):
        """Build the generator model"""
        # Input layers
        noise_input = layers.Input(shape=(self.latent_dim,), name='noise_input')
        condition_input = layers.Input(shape=(self.input_window, self.n_features), name='condition_input')
        
        # Flatten the condition input
        flat_condition = layers.Flatten()(condition_input)
        
        # Process noise with condition
        noise_dense = layers.Dense(128)(noise_input)
        noise_act = layers.LeakyReLU(alpha=0.2)(noise_dense)
        
        # Process condition
        cond_dense = layers.Dense(128)(flat_condition)
        cond_act = layers.LeakyReLU(alpha=0.2)(cond_dense)
        
        # Combine noise and condition
        combined = layers.Concatenate()([noise_act, cond_act])
        
        # Dense layers
        x = layers.Dense(256)(combined)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Reshape to prepare for convolutional layers
        reshaped = layers.Dense(self.prediction_horizon * self.n_features * 4)(x)
        reshaped = layers.LeakyReLU(alpha=0.2)(reshaped)
        reshaped = layers.Reshape((self.prediction_horizon, self.n_features * 4))(reshaped)
        
        # Conv1D layers for temporal patterns
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same')(reshaped)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        output = layers.Conv1D(filters=self.n_features, kernel_size=1, activation='sigmoid', padding='same')(x)
        
        return models.Model([noise_input, condition_input], output, name='generator')
    
    def _build_discriminator(self):
        """Build the discriminator model"""
        # Input layers
        input_sequence = layers.Input(shape=(self.input_window, self.n_features), name='input_sequence')
        predicted_sequence = layers.Input(shape=(self.prediction_horizon, self.n_features), name='predicted_sequence')
        
        # Process input sequence with Conv1D
        x1 = layers.Conv1D(filters=64, kernel_size=3, padding='same')(input_sequence)
        x1 = layers.LeakyReLU(alpha=0.2)(x1)
        x1 = layers.Dropout(0.3)(x1)
        
        x1 = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x1)
        x1 = layers.LeakyReLU(alpha=0.2)(x1)
        x1 = layers.Dropout(0.3)(x1)
        
        # Process predicted sequence with Conv1D
        x2 = layers.Conv1D(filters=64, kernel_size=3, padding='same')(predicted_sequence)
        x2 = layers.LeakyReLU(alpha=0.2)(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        x2 = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x2)
        x2 = layers.LeakyReLU(alpha=0.2)(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Flatten both
        x1 = layers.Flatten()(x1)
        x2 = layers.Flatten()(x2)
        
        # Concatenate
        x = layers.Concatenate()([x1, x2])
        
        # Dense layers
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model([input_sequence, predicted_sequence], output, name='discriminator')
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
        return model
    
    def _build_predictive_model(self):
        """Build a separate non-GAN predictive model using LSTM for evaluation purposes"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.input_window, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(128),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.prediction_horizon * self.n_features, activation='sigmoid'),
            layers.Reshape((self.prediction_horizon, self.n_features))
        ])
        
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model
    
    def _build_gan(self):
        """Build the GAN by combining generator and discriminator"""
        # Ensure discriminator is trainable before creating GAN
        self.discriminator.trainable = True
        
        # GAN inputs
        noise_input = layers.Input(shape=(self.latent_dim,), name='gan_noise_input')
        condition_input = layers.Input(shape=(self.input_window, self.n_features), name='gan_condition_input')
        
        # Generator output
        generated_sequence = self.generator([noise_input, condition_input])
        
        # Freeze the discriminator for GAN training (AFTER it's been used in the model)
        self.discriminator.trainable = False
        
        # Discriminator output
        validity = self.discriminator([condition_input, generated_sequence])
        
        # Combined GAN model
        model = models.Model([noise_input, condition_input], validity)
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def _create_sequences(self, data):
        """Create input-output sequences for training"""
        X, y = [], []
        
        # Assume data is scaled already
        for i in range(len(data) - self.input_window - self.prediction_horizon + 1):
            X.append(data[i:i+self.input_window])
            y.append(data[i+self.input_window:i+self.input_window+self.prediction_horizon])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=100, validation_split=0.2, early_stopping_patience=10):
        """
        Train the GAN model on the spectrum demand data
        
        Args:
            data: DataFrame with Time column and spectrum bands
            epochs: Number of epochs to train
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            self: Trained model
        """
        # Extract relevant columns (exclude Time)
        data_values = data.drop(columns=['Time']).values
        self.n_features = data_values.shape[1]
        
        # Scale the data
        scaled_data = self.input_scaler.fit_transform(data_values)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Build models if not already built
        if self.generator is None:
            print("Building generator...")
            self.generator = self._build_generator()
            
        if self.discriminator is None:
            print("Building discriminator...")
            self.discriminator = self._build_discriminator()
            
        if self.gan is None:
            print("Building GAN...")
            self.gan = self._build_gan()
            
        if self.predictive_model is None:
            print("Building predictive model for evaluation...")
            self.predictive_model = self._build_predictive_model()
        
        # Train the predictive model for comparison
        print("Training predictive model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        self.predictive_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        # Train the GAN
        print("Training GAN...")
        d_losses, g_losses = [], []
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of sequences
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            input_seqs = X_train[idx]
            real_future_seqs = y_train[idx]
            
            # Generate a batch of new future sequences
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_future_seqs = self.generator.predict([noise, input_seqs], verbose=0)
            
            # Make sure discriminator is trainable
            self.discriminator.trainable = True
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([input_seqs, real_future_seqs], valid)
            d_loss_fake = self.discriminator.train_on_batch([input_seqs, gen_future_seqs], fake)
            
            # Calculate average loss and accuracy
            d_loss = [0.5 * (d_loss_real[0] + d_loss_fake[0]), 0.5 * (d_loss_real[1] + d_loss_fake[1])]
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Make sure discriminator is not trainable for generator training
            self.discriminator.trainable = False
            
            # Train the generator to fool the discriminator
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch([noise, input_seqs], valid)
            
            # Save losses
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            
            # Print progress - FIX: Ensure d_loss and g_loss are properly formatted
            if epoch % 10 == 0:
                #print(f"Epoch {epoch}/{epochs}, [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.1f}%] [G loss: {g_loss:.4f}]")
                print(f"Epoch {epoch}/{epochs}")
                
                # Validate on a small subset
                val_idx = np.random.randint(0, X_val.shape[0], min(self.batch_size, X_val.shape[0]))
                val_input_seqs = X_val[val_idx]
                val_real_future_seqs = y_val[val_idx]
                
                # Generate predictions
                noise = np.random.normal(0, 1, (len(val_idx), self.latent_dim))
                val_gen_future_seqs = self.generator.predict([noise, val_input_seqs], verbose=0)
                
                # Calculate validation MSE
                val_mse = np.mean((val_real_future_seqs - val_gen_future_seqs) ** 2)
                print(f"Validation MSE: {val_mse:.4f}")
        
        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        
        # Create output directory if it doesn't exist
        os.makedirs('gan_results', exist_ok=True)
        plt.savefig('gan_results/training_losses.png')
        
        return self
    
    def predict_future(self, input_sequence, num_predictions=1, include_history=True):
        """
        Predict future spectrum demand
        
        Args:
            input_sequence: Input time sequence (with shape [1, input_window, n_features])
            num_predictions: Number of predictions to make
            include_history: Whether to include the input sequence in the result
            
        Returns:
            DataFrame with predicted future demand
        """
        # Check input shape
        if len(input_sequence.shape) == 2:
            # Add batch dimension if missing
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        # Scale the input if needed
        if input_sequence.max() > 1 or input_sequence.min() < 0:
            # Reshape to 2D for scaling
            orig_shape = input_sequence.shape
            input_sequence = input_sequence.reshape(-1, self.n_features)
            input_sequence = self.input_scaler.transform(input_sequence)
            input_sequence = input_sequence.reshape(orig_shape)
        
        # Start with the input sequence
        current_sequence = input_sequence.copy()
        all_predictions = []
        
        # Optionally include the history
        if include_history:
            # Reshape to 2D for inverse scaling
            hist_data = current_sequence.reshape(-1, self.n_features)
            hist_data = self.input_scaler.inverse_transform(hist_data)
            
            # Store each time step
            for i in range(hist_data.shape[0]):
                all_predictions.append(hist_data[i])
        
        # Generate predictions
        for _ in range(num_predictions):
            # Generate noise
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            
            # Predict next sequence
            gen_future = self.generator.predict([noise, current_sequence], verbose=0)
            
            # Store predictions
            pred_data = gen_future.reshape(-1, self.n_features)
            pred_data = self.input_scaler.inverse_transform(pred_data)
            
            for i in range(pred_data.shape[0]):
                all_predictions.append(pred_data[i])
            
            # Update current sequence for next prediction (slide window)
            current_sequence = np.roll(current_sequence, -self.prediction_horizon, axis=1)
            current_sequence[0, -self.prediction_horizon:, :] = gen_future[0]
        
        # Convert to dataframe
        columns = [f'Band_{i+1}' for i in range(self.n_features)]
        df_pred = pd.DataFrame(all_predictions, columns=columns)
        
        # Create time index
        start_date = datetime.now()
        step_size = 1  # hours
        
        if include_history:
            start_date = start_date - timedelta(hours=self.input_window*step_size)
            
        time_index = [start_date + timedelta(hours=i*step_size) for i in range(len(df_pred))]
        df_pred['Time'] = time_index
        
        return df_pred
    
    def evaluate(self, test_data):
        """
        Evaluate both GAN and predictive model on test data
        
        Args:
            test_data: DataFrame with Time column and spectrum bands
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Extract relevant columns (exclude Time)
        test_values = test_data.drop(columns=['Time']).values
        
        # Scale the data
        scaled_data = self.input_scaler.transform(test_values)
        
        # Create sequences
        X_test, y_test = self._create_sequences(scaled_data)
        
        # Evaluate predictive model
        pred_model_mse = self.predictive_model.evaluate(X_test, y_test, verbose=0)[0]
        
        # Evaluate GAN
        gan_mse = 0
        for i in range(0, len(X_test), self.batch_size):
            batch_X = X_test[i:i+self.batch_size]
            batch_y = y_test[i:i+self.batch_size]
            
            if len(batch_X) == 0:
                continue
                
            # Generate noise
            noise = np.random.normal(0, 1, (len(batch_X), self.latent_dim))
            
            # Generate predictions
            gen_future = self.generator.predict([noise, batch_X], verbose=0)
            
            # Calculate MSE
            batch_mse = np.mean((batch_y - gen_future) ** 2)
            gan_mse += batch_mse * len(batch_X)
            
        gan_mse /= len(X_test)
        
        # Compare both models on a sample
        sample_idx = np.random.randint(0, len(X_test))
        sample_X = X_test[sample_idx:sample_idx+1]
        sample_y = y_test[sample_idx:sample_idx+1]
        
        # Get predictions from both models
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gan_pred = self.generator.predict([noise, sample_X], verbose=0)
        pred_model_pred = self.predictive_model.predict(sample_X, verbose=0)
        
        # Inverse transform
        sample_X_inv = self.input_scaler.inverse_transform(sample_X.reshape(-1, self.n_features))
        sample_X_inv = sample_X_inv.reshape(sample_X.shape)
        
        sample_y_inv = self.input_scaler.inverse_transform(sample_y.reshape(-1, self.n_features))
        sample_y_inv = sample_y_inv.reshape(sample_y.shape)
        
        gan_pred_inv = self.input_scaler.inverse_transform(gan_pred.reshape(-1, self.n_features))
        gan_pred_inv = gan_pred_inv.reshape(gan_pred.shape)
        
        pred_model_pred_inv = self.input_scaler.inverse_transform(pred_model_pred.reshape(-1, self.n_features))
        pred_model_pred_inv = pred_model_pred_inv.reshape(pred_model_pred.shape)
        
        # Plot comparison
        plt.figure(figsize=(15, 10))
        
        # Get the band names or create generic ones
        band_names = [f'Band_{i+1}' for i in range(self.n_features)]
        
        for i, band in enumerate(band_names):
            plt.subplot(len(band_names), 1, i+1)
            
            # Plot history
            history_x = np.arange(self.input_window)
            plt.plot(history_x, sample_X_inv[0, :, i], 'k-', label='History')
            
            # Plot real future
            future_x = np.arange(self.input_window, self.input_window + self.prediction_horizon)
            plt.plot(future_x, sample_y_inv[0, :, i], 'g-', label='Real Future')
            
            # Plot GAN prediction
            plt.plot(future_x, gan_pred_inv[0, :, i], 'r--', label='GAN Prediction')
            
            # Plot normal prediction model
            plt.plot(future_x, pred_model_pred_inv[0, :, i], 'b--', label='LSTM Prediction')
            
            plt.title(f'{band} Demand Prediction')
            plt.ylabel('Demand (%)')
            plt.legend()
            
            if i == len(band_names) - 1:
                plt.xlabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig('gan_results/prediction_comparison.png')
        
        return {
            'gan_mse': gan_mse,
            'predictive_model_mse': pred_model_mse,
            'sample_plot': 'prediction_comparison.png'
        }
    
    def save_models(self, model_dir='gan_spectrum_models'):
        """Save the models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save generator
        self.generator.save(f'{model_dir}/generator.h5')
        
        # Save discriminator
        self.discriminator.save(f'{model_dir}/discriminator.h5')
        
        # Save predictive model
        self.predictive_model.save(f'{model_dir}/predictive_model.h5')
        
        # Save scalers - properly save the attributes
        if hasattr(self.input_scaler, 'data_range_'):
            np.save(f'{model_dir}/input_scaler_data_range.npy', self.input_scaler.data_range_)
            np.save(f'{model_dir}/input_scaler_data_min.npy', self.input_scaler.data_min_)
            np.save(f'{model_dir}/input_scaler_scale.npy', self.input_scaler.scale_)
            np.save(f'{model_dir}/input_scaler_min.npy', self.input_scaler.min_)
        
        if hasattr(self.output_scaler, 'data_range_'):
            np.save(f'{model_dir}/output_scaler_data_range.npy', self.output_scaler.data_range_)
            np.save(f'{model_dir}/output_scaler_data_min.npy', self.output_scaler.data_min_)
            np.save(f'{model_dir}/output_scaler_scale.npy', self.output_scaler.scale_)
            np.save(f'{model_dir}/output_scaler_min.npy', self.output_scaler.min_)
        
        print(f"Models saved to {model_dir}/")

    @classmethod
    def load_models(cls, model_dir='gan_spectrum_models', input_window=24, prediction_horizon=6):
        """Load the models from disk"""
        predictor = cls(input_window=input_window, prediction_horizon=prediction_horizon)
        
        # Load generator
        predictor.generator = models.load_model(f'{model_dir}/generator.h5')
        
        # Load discriminator
        predictor.discriminator = models.load_model(f'{model_dir}/discriminator.h5')
        
        # Load predictive model
        predictor.predictive_model = models.load_model(f'{model_dir}/predictive_model.h5')
        
        # Set n_features from generator output shape
        predictor.n_features = predictor.generator.output_shape[-1]
        
        # Load input scaler
        if os.path.exists(f'{model_dir}/input_scaler_data_range.npy'):
            predictor.input_scaler = MinMaxScaler()
            predictor.input_scaler.data_range_ = np.load(f'{model_dir}/input_scaler_data_range.npy')
            predictor.input_scaler.data_min_ = np.load(f'{model_dir}/input_scaler_data_min.npy')
            predictor.input_scaler.scale_ = np.load(f'{model_dir}/input_scaler_scale.npy')
            predictor.input_scaler.min_ = np.load(f'{model_dir}/input_scaler_min.npy')
            
        # Load output scaler
        if os.path.exists(f'{model_dir}/output_scaler_data_range.npy'):
            predictor.output_scaler = MinMaxScaler()
            predictor.output_scaler.data_range_ = np.load(f'{model_dir}/output_scaler_data_range.npy')
            predictor.output_scaler.data_min_ = np.load(f'{model_dir}/output_scaler_data_min.npy')
            predictor.output_scaler.scale_ = np.load(f'{model_dir}/output_scaler_scale.npy')
            predictor.output_scaler.min_ = np.load(f'{model_dir}/output_scaler_min.npy')
        
        # Rebuild GAN
        predictor.gan = predictor._build_gan()
        
        print(f"Models loaded from {model_dir}/")
        return predictor

# Sample data generation function for testing
def generate_sample_data(days=60, bands=5, hourly=True):
    """Generate sample spectrum demand data for testing"""
    # Set number of samples
    if hourly:
        samples = days * 24
    else:
        samples = days
        
    # Create time index
    start_date = datetime.now() - timedelta(days=days)
    if hourly:
        time_index = [start_date + timedelta(hours=i) for i in range(samples)]
    else:
        time_index = [start_date + timedelta(days=i) for i in range(samples)]
    
    # Create dataframe
    df = pd.DataFrame()
    df['Time'] = time_index
    
    # Generate band data with daily and weekly patterns
    for i in range(bands):
        # Base demand with some randomness
        base = 40 + np.random.normal(0, 5, samples)
        
        # Daily pattern (higher during day, lower at night)
        if hourly:
            daily = 15 * np.sin(np.pi * np.arange(samples) % 24 / 12)
        else:
            daily = np.zeros(samples)
            
        # Weekly pattern (higher on weekdays)
        if hourly:
            day_of_week = np.array([d.weekday() for d in time_index])
            weekly = 10 * (day_of_week < 5)
        else:
            weekly = np.zeros(samples)
            
        # Long term trend (slight increase or decrease)
        trend = np.linspace(0, 5 * np.random.choice([-1, 1]), samples)
        
        # Combine patterns
        demand = base + daily + weekly + trend
        
        # Ensure values are between 0 and 100
        demand = np.clip(demand, 0, 100)
        
        # Add to dataframe
        df[f'Band_{i+1}'] = demand
    
    return df


# Example main function to demonstrate usage
def main():
    """
    Main function to demonstrate GAN-based spectrum demand prediction
    """
    # Assume we have generated data with TimeGAN as shown in your code
    data_dir = "spectrum_demand_data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Find CSV files
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"No data files found in {data_dir}. Generating sample data...")
        
        # Generate some sample data
        sample_df = generate_sample_data(days=60)
        
        # Save sample data
        sample_path = f"{data_dir}/sample_spectrum_data.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample data saved to {sample_path}")
        
        # Split into train and test
        train_data = sample_df.iloc[:1200]  # First 50 days
        test_data = sample_df.iloc[1200:]   # Last 10 days
    else:
        # Load the first file for demonstration
        print(f"Loading data from {csv_files[0]}...")
        df = pd.read_csv(csv_files[0])
        
        # Convert Time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])
        
        # Split into train and test sets
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
    
    # Initialize predictor
    print("Initializing GAN-based spectrum demand predictor...")
    predictor = SpectrumDemandPredictor(
        input_window=24,       # Use 24 hours of history
        prediction_horizon=12, # Predict 12 hours ahead
        latent_dim=100,        # Dimension of latent space
        batch_size=64          # Batch size for training
    )
    
    # Train the models - reduce epochs for quick testing
    print("Training models...")
    predictor.fit(
        train_data,
        epochs=50,                   # Reduced for testing
        validation_split=0.2,         # Validation split
        early_stopping_patience=10    # Early stopping patience
    )
    
    # Evaluate on test data
    print("Evaluating models...")
    eval_metrics = predictor.evaluate(test_data)
    print(f"GAN MSE: {eval_metrics['gan_mse']:.4f}")
    print(f"Predictive Model MSE: {eval_metrics['predictive_model_mse']:.4f}")
    
    # Make predictions for the next 3 days (72 hours)
    print("Generating predictions for the next 3 days...")
    input_seq = test_data.drop(columns=['Time']).values[:24]  # Use first 24 hours of test data
    
    # Get predictions
    predictions = predictor.predict_future(
        input_seq,
        num_predictions=6,  # Predict 6 windows ahead (6 x 12 = 72 hours)
        include_history=True
    )
    
    # Plot the predictions
    plt.figure(figsize=(15, 10))
    
    # Get the band names
    band_names = [col for col in train_data.columns if col != 'Time']
    
    for i, band in enumerate(band_names):
        plt.subplot(len(band_names), 1, i+1)
        plt.plot(predictions['Time'], predictions[f'Band_{i+1}'], 'b-', label='Prediction')
        plt.title(f'{band} Demand Prediction')
        plt.ylabel('Demand (%)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('gan_results/future_predictions.png')
    
    # Save the trained models
    print("Saving models...")
    predictor.save_models()
    
    print("Done! Check the 'gan_results' directory for plots and 'gan_spectrum_models' for saved models.")


if __name__ == "__main__":
    main()