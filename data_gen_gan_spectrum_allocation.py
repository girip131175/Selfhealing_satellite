# Load the libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import random
import os
from datetime import datetime, timedelta

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print('Using GPU')
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
else:
    print('Using CPU')

# Define frequency bands for 4G and 5G
BANDS = {
    '4G_Low': {'range': '700-900 MHz', 'bandwidth': 20},    # Low-band 4G
    '4G_Mid': {'range': '1.8-2.1 GHz', 'bandwidth': 40},    # Mid-band 4G
    '4G_High': {'range': '2.3-2.6 GHz', 'bandwidth': 60},   # High-band 4G
    '5G_Low': {'range': '600-700 MHz', 'bandwidth': 50},    # Low-band 5G
    '5G_Mid': {'range': '2.5-3.7 GHz', 'bandwidth': 100},   # Mid-band 5G (C-band)
    '5G_High': {'range': '24-40 GHz', 'bandwidth': 400}     # High-band 5G (mmWave)
}

# Function to generate synthetic hourly data if real data is not available
def generate_sample_data(days=30, bands=BANDS.keys()):
    """
    Generate sample hourly spectrum data with additional parameters
    """
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=h) for h in range(24 * days)]
    
    data = {'Time': dates}
    
    for band in bands:
        # Base parameters from original code
        if 'Low' in band:
            base = 80
            amplitude = 15
        elif 'Mid' in band:
            base = 60
            amplitude = 20
        else:  # High
            base = 40
            amplitude = 25
            
        # Generate base demand pattern
        if '4G' in band:
            hourly_pattern = np.array([
                0.6, 0.5, 0.4, 0.3, 0.3, 0.4,
                0.6, 0.8, 0.9, 0.8, 0.7, 0.9,
                1.0, 0.9, 0.8, 0.9, 1.0, 1.1,
                1.2, 1.1, 1.0, 0.9, 0.8, 0.7
            ])
            noise_level = 0.05
        else:
            hourly_pattern = np.array([
                0.4, 0.3, 0.2, 0.2, 0.2, 0.3,
                0.5, 0.7, 0.9, 1.0, 1.1, 1.2,
                1.3, 1.2, 1.1, 1.2, 1.4, 1.5,
                1.3, 1.1, 0.9, 0.7, 0.6, 0.5
            ])
            noise_level = 0.1

        # Initialize lists for new parameters
        demand = []
        snr = []
        rssi = []
        active_users = []
        data_rate = []
        traffic_load = []
        interference = []
        tx_power = []
        
        for i, date in enumerate(dates):
            hour = date.hour
            day = date.weekday()
            
            # Base demand calculation (from original code)
            value = base + amplitude * hourly_pattern[hour]
            if day >= 5:
                if '5G' in band and 'High' in band:
                    value *= 0.7
                elif 'Mid' in band:
                    value *= 0.9
                else:
                    value *= 0.85
            
            value += np.random.normal(0, noise_level * amplitude)
            value = max(0, min(100, value))
            
            # Generate new parameters based on demand
            users = int(np.random.normal(value * 5, value * 0.5))  # Active users
            snr_val = 25 + np.random.normal(0, 3) - (users / 100)  # SNR (dB)
            rssi_val = -70 - (users / 50) + np.random.normal(0, 2)  # RSSI (dBm)
            data_rate_val = BANDS[band]['bandwidth'] * (value/100) * (1 + np.random.normal(0, 0.1))
            interference_val = -90 + (users / 30) + np.random.normal(0, 3)
            tx_power_val = 20 + np.random.normal(0, 1)  # Transmission power (dBm)
            
            # Append all values
            demand.append(value)
            snr.append(snr_val)
            rssi.append(rssi_val)
            active_users.append(users)
            data_rate.append(data_rate_val)
            traffic_load.append(value)  # Using original demand as traffic load
            interference.append(interference_val)
            tx_power.append(tx_power_val)
        
        # Add all parameters to the dataset
        data[f"{band}_Demand"] = demand
        data[f"{band}_SNR"] = snr
        data[f"{band}_RSSI"] = rssi
        data[f"{band}_Users"] = active_users
        data[f"{band}_DataRate"] = data_rate
        data[f"{band}_Traffic"] = traffic_load
        data[f"{band}_Interference"] = interference
        data[f"{band}_TxPower"] = tx_power
    
    df = pd.DataFrame(data)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

# TimeGAN class to handle generation
class SpectrumTimeGAN:
    def __init__(self, seq_len=24, hidden_dim=24, num_layers=3, batch_size=128, 
                 train_steps=10000, gamma=1.0):
        """
        Initialize the TimeGAN model for spectrum data
        
        Args:
            seq_len: sequence length for the LSTM/GRU
            hidden_dim: dimension of hidden layers
            num_layers: number of layers in RNNs
            batch_size: batch size for training
            train_steps: number of training steps
            gamma: discount factor for discriminator loss
        """
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.gamma = gamma
        self.n_seq = None  # Will be set based on input data
        
        # Create TF models
        self.embedder = None
        self.recovery = None
        self.generator = None
        self.discriminator = None
        self.supervisor = None
        
        # Create TF optimizers
        self.autoencoder_optimizer = Adam()
        self.supervisor_optimizer = Adam()
        self.generator_optimizer = Adam()
        self.discriminator_optimizer = Adam()
        self.embedding_optimizer = Adam()
        
        # Loss functions
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
    
    def make_rnn(self, n_layers, hidden_units, output_units, name):
        """Create a RNN model with the specified architecture"""
        return Sequential([
            GRU(units=hidden_units, return_sequences=True, name=f'{name}_GRU_{i+1}') 
            for i in range(n_layers)] + 
            [Dense(units=output_units, activation='sigmoid', name=f'{name}_OUT')], 
            name=name)
    
    def fit(self, data):
        """
        Train the TimeGAN model on the given spectrum data
        
        Args:
            data: DataFrame with Time column and spectrum bands
        
        Returns:
            self: Trained model
        """
        # Remove Time column for training
        train_data = data.drop(columns=['Time']).values
        self.n_seq = train_data.shape[1]
        
        # Normalize data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(train_data).astype(np.float32)
        
        # Create sequential windows
        windows = []
        for i in range(len(scaled_data) - self.seq_len):
            windows.append(scaled_data[i:i + self.seq_len])
        
        n_windows = len(windows)
        print(f"Created {n_windows} training sequences")
        
        # Create TF datasets
        real_series = (tf.data.Dataset
                      .from_tensor_slices(windows)
                      .shuffle(buffer_size=n_windows)
                      .batch(self.batch_size))
        self.real_series_iter = iter(real_series.repeat())
        
        # Random data generator
        def make_random_data():
            while True:
                yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))
        
        self.random_series = iter(tf.data.Dataset
                                 .from_generator(make_random_data, output_types=tf.float32)
                                 .batch(self.batch_size)
                                 .repeat())
        
        # Initialize network architectures
        print("Building TimeGAN components...")
        self._build_networks()
        
        # Phase 1: Train autoencoder
        print("Phase 1: Training autoencoder...")
        self._train_autoencoder()
        
        # Phase 2: Train supervisor
        print("Phase 2: Training supervisor...")
        self._train_supervisor()
        
        # Phase 3: Joint Training
        print("Phase 3: Joint training of generator and discriminator...")
        self._joint_train()
        
        return self
    
    def _build_networks(self):
        """Build all network components of TimeGAN"""
        # Input placeholders
        self.X = Input(shape=[self.seq_len, self.n_seq], name='RealData')
        self.Z = Input(shape=[self.seq_len, self.n_seq], name='RandomData')
        
        # Build networks
        self.embedder = self.make_rnn(self.num_layers, self.hidden_dim, self.hidden_dim, 'Embedder')
        self.recovery = self.make_rnn(self.num_layers, self.hidden_dim, self.n_seq, 'Recovery')
        self.generator = self.make_rnn(self.num_layers, self.hidden_dim, self.hidden_dim, 'Generator')
        self.discriminator = self.make_rnn(self.num_layers, self.hidden_dim, 1, 'Discriminator')
        self.supervisor = self.make_rnn(self.num_layers-1, self.hidden_dim, self.hidden_dim, 'Supervisor')
        
        # Build the autoencoder
        H = self.embedder(self.X)
        X_tilde = self.recovery(H)
        self.autoencoder = Model(inputs=self.X, outputs=X_tilde, name='Autoencoder')
        
        # Build adversarial networks
        E_hat = self.generator(self.Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)
        self.adversarial_supervised = Model(inputs=self.Z, outputs=Y_fake, name='AdversarialNetSupervised')
        
        # Latent space adversarial
        Y_fake_e = self.discriminator(E_hat)
        self.adversarial_emb = Model(inputs=self.Z, outputs=Y_fake_e, name='AdversarialNet')
        
        # Synthetic data generator
        X_hat = self.recovery(H_hat)
        self.synthetic_data = Model(inputs=self.Z, outputs=X_hat, name='SyntheticData')
        
        # Discriminator for real data
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=self.X, outputs=Y_real, name='DiscriminatorReal')
    
    @tf.function
    def _train_autoencoder_step(self, x):
        """Single training step for autoencoder"""
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)
        
        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        self.autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)
    
    def _train_autoencoder(self):
        """Train the autoencoder"""
        for step in tqdm(range(self.train_steps)):
            X_ = next(self.real_series_iter)
            step_e_loss_t0 = self._train_autoencoder_step(X_)
            
            if step % 1000 == 0:
                print(f'Autoencoder step: {step}, loss: {step_e_loss_t0:.4f}')
    
    @tf.function
    def _train_supervisor_step(self, x):
        """Single training step for supervisor"""
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            g_loss_s = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])
        
        var_list = self.supervisor.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        self.supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s
    
    def _train_supervisor(self):
        """Train the supervisor"""
        for step in tqdm(range(self.train_steps)):
            X_ = next(self.real_series_iter)
            step_g_loss_s = self._train_supervisor_step(X_)
            
            if step % 1000 == 0:
                print(f'Supervisor step: {step}, loss: {step_g_loss_s:.4f}')
    
    def _get_generator_moment_loss(self, y_true, y_pred):
        """Calculate the moment loss for the generator"""
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var
    
    @tf.function
    def _train_generator_step(self, x, z):
        """Single training step for generator"""
        with tf.GradientTape() as tape:
            y_fake = self.adversarial_supervised(z)
            generator_loss_unsupervised = self.bce(y_true=tf.ones_like(y_fake),
                                                  y_pred=y_fake)
            
            y_fake_e = self.adversarial_emb(z)
            generator_loss_unsupervised_e = self.bce(y_true=tf.ones_like(y_fake_e),
                                                    y_pred=y_fake_e)
            
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
            
            x_hat = self.synthetic_data(z)
            generator_moment_loss = self._get_generator_moment_loss(x, x_hat)
            
            generator_loss = (generator_loss_unsupervised +
                             generator_loss_unsupervised_e +
                             100 * tf.sqrt(generator_loss_supervised) +
                             100 * generator_moment_loss)
        
        var_list = self.generator.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        self.generator_optimizer.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
    
    @tf.function
    def _train_embedder_step(self, x):
        """Single training step for embedder"""
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
            
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised
        
        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        self.embedding_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)
    
    @tf.function
    def _get_discriminator_loss(self, x, z):
        """Calculate discriminator loss"""
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self.bce(y_true=tf.ones_like(y_real),
                                          y_pred=y_real)
        
        y_fake = self.adversarial_supervised(z)
        discriminator_loss_fake = self.bce(y_true=tf.zeros_like(y_fake),
                                          y_pred=y_fake)
        
        y_fake_e = self.adversarial_emb(z)
        discriminator_loss_fake_e = self.bce(y_true=tf.zeros_like(y_fake_e),
                                            y_pred=y_fake_e)
        
        return (discriminator_loss_real +
               discriminator_loss_fake +
               self.gamma * discriminator_loss_fake_e)
    
    @tf.function
    def _train_discriminator_step(self, x, z):
        """Single training step for discriminator"""
        with tf.GradientTape() as tape:
            discriminator_loss = self._get_discriminator_loss(x, z)
        
        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        self.discriminator_optimizer.apply_gradients(zip(gradients, var_list))
        return discriminator_loss
    
    def _joint_train(self):
        """Joint training of generator and discriminator"""
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for step in tqdm(range(self.train_steps)):
            # Train generator (twice as often as discriminator)
            for _ in range(2):
                X_ = next(self.real_series_iter)
                Z_ = next(self.random_series)
                
                # Train generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self._train_generator_step(X_, Z_)
                # Train embedder
                step_e_loss_t0 = self._train_embedder_step(X_)
            
            X_ = next(self.real_series_iter)
            Z_ = next(self.random_series)
            step_d_loss = self._get_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = self._train_discriminator_step(X_, Z_)
            
            if step % 1000 == 0:
                print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                     f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')
    
    def generate(self, n_samples, time_index):
        """
        Generate synthetic spectrum demand data
        
        Args:
            n_samples: Number of synthetic sequences to generate
            time_index: Time index for the generated data
        
        Returns:
            DataFrame with synthetic data
        """
        print(f"Generating {n_samples} synthetic spectrum demand sequences...")
        generated_data = []
        
        # Generate synthetic sequences
        for _ in range(int(np.ceil(n_samples / self.batch_size))):
            Z_ = next(self.random_series)
            d = self.synthetic_data(Z_)
            generated_data.append(d)
        
        # Stack and reshape
        generated_data = np.array(np.vstack(generated_data))
        
        # Inverse transform to get original scale
        generated_data = (self.scaler.inverse_transform(
            generated_data.reshape(-1, self.n_seq))
            .reshape(-1, self.seq_len, self.n_seq))
        
        # Sample the required number of sequences
        if len(generated_data) > n_samples:
            indices = random.sample(range(len(generated_data)), n_samples)
            generated_data = generated_data[indices]
        
        # Flatten and create DataFrame
        flat_data = generated_data.reshape(-1, self.n_seq)
        
        # Use provided time index or create new one
        if time_index is None or len(time_index) != len(flat_data):
            start_date = datetime(2023, 1, 1)
            time_index = [start_date + timedelta(hours=h) for h in range(len(flat_data))]
        
        # Create column names for all parameters
        columns = []
        for band in BANDS.keys():
            for param in ['Demand', 'SNR', 'RSSI', 'Users', 'DataRate', 'Traffic', 'Interference', 'TxPower']:
                columns.append(f"{band}_{param}")
        
        # Create DataFrame with all columns
        df = pd.DataFrame(flat_data, columns=columns)
        df['Time'] = time_index[:len(df)]
        
        # Ensure valid values (0-100%) for Demand and Traffic
        for band in BANDS.keys():
            df[f"{band}_Demand"] = df[f"{band}_Demand"].clip(0, 100)
            df[f"{band}_Traffic"] = df[f"{band}_Traffic"].clip(0, 100)
        
        return df

# Main execution function
def generate_spectrum_demand_dataset(sample_size=30, seq_length=24, num_iterations=5, 
                                   output_dir="spectrum_demand_data"):
    """
    Generate synthetic spectrum demand data across different frequency bands
    
    Args:
        sample_size: Number of days to generate in each iteration
        seq_length: Sequence length for TimeGAN (typically 24 for daily patterns)
        num_iterations: Number of different datasets to generate
        output_dir: Directory to save output files
    """
    print(f"Generating synthetic spectrum demand data with TimeGAN")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate or load initial sample data
    print("Generating initial sample data...")
    sample_df = generate_sample_data(days=sample_size)
    
    # Set up TimeGAN model
    model = SpectrumTimeGAN(
        seq_len=seq_length,
        hidden_dim=32,
        num_layers=3,
        batch_size=64,
        train_steps=5000  # Reduced for demonstration, use 10000+ for production
    )
    
    # Train the model
    print("Training TimeGAN model on spectrum data...")
    model.fit(sample_df)
    
    # Generate synthetic datasets
    for i in range(num_iterations):
        print(f"Generating synthetic dataset {i+1}/{num_iterations}")
        
        # Generate synthetic data (approximately 3x the original size)
        synth_df = model.generate(n_samples=sample_size*3, time_index=None)
        
        # Save the dataset
        output_file = os.path.join(output_dir, f"spectrum_demand_{i+1}.csv")
        synth_df.to_csv(output_file, index=False)
        print(f"Saved synthetic dataset to {output_file}")
        
        # Optional: Create and save visualizations
        if i == 0:  # Only for the first iteration to avoid clutter
            visualize_data(sample_df, synth_df, output_dir)
    
    print(f"Successfully generated {num_iterations} synthetic spectrum demand datasets in {output_dir}")
    return True

def visualize_data(real_df, synth_df, output_dir):
    """Enhanced visualization function with new parameters"""
    # Create separate plots for different parameter groups
    parameter_groups = {
        'spectrum': ['Demand', 'SNR', 'RSSI'],
        'traffic': ['Users', 'DataRate', 'Traffic'],
        'power': ['Interference', 'TxPower']
    }
    
    for group_name, parameters in parameter_groups.items():
        plt.figure(figsize=(15, 12))
        hours_to_plot = 168  # One week
        
        for band in BANDS.keys():
            for i, param in enumerate(parameters):
                plt.subplot(len(parameters), 1, i+1)
                
                column = f"{band}_{param}"
                if column in real_df.columns:
                    plt.plot(real_df['Time'][:hours_to_plot], 
                            real_df[column][:hours_to_plot], 
                            label=f'{band} (Real)', alpha=0.7)
                    plt.plot(synth_df['Time'][:hours_to_plot], 
                            synth_df[column][:hours_to_plot], 
                            label=f'{band} (Synth)', alpha=0.7, linestyle='--')
                
                plt.title(f'{param} Over Time')
                plt.ylabel(get_parameter_unit(param))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectrum_{group_name}_comparison.png'), 
                   bbox_inches='tight')
        plt.close()

def get_parameter_unit(param):
    """Helper function to return appropriate units for parameters"""
    units = {
        'Demand': 'Utilization (%)',
        'SNR': 'dB',
        'RSSI': 'dBm',
        'Users': 'Count',
        'DataRate': 'Mbps',
        'Traffic': 'Utilization (%)',
        'Interference': 'dBm',
        'TxPower': 'dBm'
    }
    return units.get(param, '')

# Example usage
if __name__ == "__main__":
    # Generate synthetic spectrum demand data
    generate_spectrum_demand_dataset(
        sample_size=30,  # 30 days of initial data
        seq_length=24,   # 24-hour sequences (daily patterns)
        num_iterations=5,  # Generate 5 different datasets
        output_dir="spectrum_demand_data"
    )