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

# Define frequency bands for 4G and 5G (same as in demand generator)
BANDS = {
    '4G_Low': '700-900 MHz',    # Low-band 4G
    '4G_Mid': '1.8-2.1 GHz',    # Mid-band 4G
    '4G_High': '2.3-2.6 GHz',   # High-band 4G
    '5G_Low': '600-700 MHz',    # Low-band 5G
    '5G_Mid': '2.5-3.7 GHz',    # Mid-band 5G (C-band)
    '5G_High': '24-40 GHz'      # High-band 5G (mmWave)
}

# Define interference sources and their characteristics
INTERFERENCE_SOURCES = {
    'radar': {
        'affected_bands': ['4G_High', '5G_Mid', '5G_High'],
        'intensity': 'high',
        'pattern': 'intermittent',
        'description': 'Weather and military radar systems'
    },
    'wifi': {
        'affected_bands': ['4G_Mid', '4G_High', '5G_Mid'],
        'intensity': 'medium',
        'pattern': 'continuous',
        'description': 'WiFi networks in 2.4 and 5GHz bands'
    },
    'satellite': {
        'affected_bands': ['4G_Low', '5G_Low', '5G_Mid'],
        'intensity': 'medium',
        'pattern': 'periodic',
        'description': 'Satellite communication systems'
    },
    'microwave': {
        'affected_bands': ['4G_Mid', '5G_Mid'],
        'intensity': 'high',
        'pattern': 'intermittent',
        'description': 'Microwave ovens and industrial equipment'
    },
    'broadcasting': {
        'affected_bands': ['4G_Low', '5G_Low'],
        'intensity': 'low',
        'pattern': 'continuous',
        'description': 'TV and radio broadcasting'
    },
    'iot_devices': {
        'affected_bands': ['4G_Low', '4G_Mid', '5G_Low'],
        'intensity': 'low',
        'pattern': 'random',
        'description': 'IoT devices and sensors'
    },
    'adjacent_cell': {
        'affected_bands': ['4G_Low', '4G_Mid', '4G_High', '5G_Low', '5G_Mid', '5G_High'],
        'intensity': 'medium',
        'pattern': 'varying',
        'description': 'Adjacent cell interference'
    },
    'industrial': {
        'affected_bands': ['4G_Low', '5G_Mid'],
        'intensity': 'high',
        'pattern': 'scheduled',
        'description': 'Industrial machinery and equipment'
    }
}

# Define interference locations (to create spatial variation)
LOCATIONS = [
    'urban_dense', 'urban_sparse', 'suburban', 'rural', 
    'industrial_zone', 'residential', 'campus', 'highway',
    'downtown', 'airport', 'seaport', 'shopping_mall'
]

def generate_interference_sample_data(days=30, bands=BANDS.keys(), locations=LOCATIONS, seed=None):
    """
    Generate sample hourly interference data for multiple bands and locations
    
    Args:
        days: Number of days to generate data for
        bands: Frequency bands to include
        locations: Locations to generate data for
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with interference data
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create a time series for the specified number of days
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=h) for h in range(24 * days)]
    
    # Create list to hold all data
    all_data = []
    
    # For each location, generate interference patterns
    for location in locations:
        # Base data with time
        data = {'Time': dates, 'Location': [location] * len(dates)}
        
        # Determine location characteristics
        is_urban = 'urban' in location or 'downtown' in location
        is_industrial = 'industrial' in location
        is_dense = 'dense' in location or 'downtown' in location or 'mall' in location
        is_transportation = 'airport' in location or 'seaport' in location or 'highway' in location
        
        # Generate interference patterns for each band
        for band in bands:
            # Base interference levels based on band and location
            if 'Low' in band:
                # Low bands have better penetration but more legacy interference
                base = 5 if is_urban else 3
                amplitude = 8 if is_urban else 5
            elif 'Mid' in band:
                # Mid bands have moderate interference, especially in dense areas
                base = 7 if is_dense else 4
                amplitude = 10 if is_dense else 6
            else:  # High bands
                # High bands have atmospheric and line-of-sight issues
                base = 9 if is_transportation or is_urban else 5
                amplitude = 12 if is_transportation or is_urban else 7
            
            # Add industrial noise for specific bands
            if is_industrial and ('4G_Mid' in band or '5G_Mid' in band):
                base += 5
                amplitude += 5
            
            # 5G specific considerations
            if '5G' in band:
                if '5G_High' in band:
                    # mmWave has high atmospheric attenuation
                    base += 4
                if is_urban and '5G_Mid' in band:
                    # C-band in urban areas has specific challenges
                    base += 3
            
            # Create sources of interference based on relevant interference types for this band
            relevant_sources = [
                source for source, props in INTERFERENCE_SOURCES.items() 
                if band in props['affected_bands']
            ]
            
            # Generate hourly pattern with multiple interference sources
            band_interference = []
            
            for date in dates:
                hour = date.hour
                day = date.weekday()
                interference_level = base
                
                # Time-of-day variations
                if 8 <= hour <= 18:  # Business hours
                    interference_level += 3 if is_urban or is_dense else 1
                if 18 <= hour <= 22:  # Evening peak
                    interference_level += 5 if is_urban or 'residential' in location else 2
                if 0 <= hour <= 5:  # Late night
                    interference_level -= 2
                
                # Day-of-week variations
                if day >= 5:  # Weekend
                    if is_urban or 'mall' in location or 'residential' in location:
                        interference_level += 2  # More residential activity
                    if is_industrial:
                        interference_level -= 3  # Less industrial activity
                
                # Add interference from relevant sources
                for source in relevant_sources:
                    source_props = INTERFERENCE_SOURCES[source]
                    source_strength = 0
                    
                    # Determine base strength by intensity
                    if source_props['intensity'] == 'high':
                        source_strength = 5
                    elif source_props['intensity'] == 'medium':
                        source_strength = 3
                    else:  # low
                        source_strength = 1
                    
                    # Apply pattern-specific variations
                    if source_props['pattern'] == 'intermittent':
                        # Randomly occurs with higher probability during certain hours
                        if random.random() < 0.3 + (0.2 if 8 <= hour <= 18 else 0):
                            source_strength *= 2
                        else:
                            source_strength = 0
                    
                    elif source_props['pattern'] == 'continuous':
                        # Always present but varies slightly
                        source_strength *= 0.8 + 0.4 * random.random()
                    
                    elif source_props['pattern'] == 'periodic':
                        # Follows a sinusoidal pattern
                        period = 8  # 8-hour cycle
                        source_strength *= 0.5 + 0.5 * np.sin(2 * np.pi * (hour % period) / period)
                    
                    elif source_props['pattern'] == 'random':
                        # Completely random
                        if random.random() < 0.4:
                            source_strength *= random.random() * 2
                        else:
                            source_strength = 0
                    
                    elif source_props['pattern'] == 'varying':
                        # Follows traffic patterns
                        if 7 <= hour <= 9 or 16 <= hour <= 19:  # Rush hours
                            source_strength *= 1.5
                        elif 22 <= hour or hour <= 5:  # Night
                            source_strength *= 0.5
                    
                    elif source_props['pattern'] == 'scheduled':
                        # Regular schedule (like industrial shifts)
                        if day < 5 and 8 <= hour <= 16:  # Weekday business hours
                            source_strength *= 1.8
                        else:
                            source_strength *= 0.3
                    
                    # Apply source strength to overall interference
                    interference_level += source_strength
                
                # Add random noise
                interference_level += np.random.normal(0, amplitude * 0.15)
                
                # Ensure values are positive and reasonable (0-100%)
                interference_level = max(0, min(100, interference_level))
                band_interference.append(interference_level)
            
            # Add to data dictionary
            data[f"{band}_interference"] = band_interference
        
        # Create DataFrame for this location
        df_location = pd.DataFrame(data)
        all_data.append(df_location)
    
    # Combine all locations
    df = pd.concat(all_data, ignore_index=True)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

from data_gen_gan_spectrum_allocation import SpectrumTimeGAN

# TimeGAN class for interference data
class InterferenceTimeGAN(SpectrumTimeGAN):
    """
    Extends the SpectrumTimeGAN class to handle interference data with location information
    """
    
    def fit(self, data, locations=None):
        """
        Train the TimeGAN model on the given interference data
        
        Args:
            data: DataFrame with Time, Location columns and interference values
            locations: Optional list of locations to process (if None, process all)
            
        Returns:
            self: Trained model
        """
        # Process data for each location separately
        if locations is None:
            locations = data['Location'].unique()
        
        self.location_models = {}
        self.location_scalers = {}
        
        for location in tqdm(locations, desc="Training location models"):
            location_data = data[data['Location'] == location].copy()
            
            # Drop non-feature columns for training
            train_cols = [col for col in location_data.columns 
                         if col not in ['Time', 'Location'] and 'interference' in col]
            
            train_data = location_data[train_cols].values
            self.n_seq = train_data.shape[1]
            
            # Normalize data for this location
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(train_data).astype(np.float32)
            self.location_scalers[location] = scaler
            
            # Create sequential windows
            windows = []
            for i in range(len(scaled_data) - self.seq_len):
                windows.append(scaled_data[i:i + self.seq_len])
            
            n_windows = len(windows)
            print(f"Location {location}: Created {n_windows} training sequences")
            
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
            print(f"Building TimeGAN components for location {location}...")
            self._build_networks()
            
            # Train the model
            print(f"Training TimeGAN for location {location}...")
            # Phase 1: Train autoencoder
            print("Phase 1: Training autoencoder...")
            self._train_autoencoder()
            
            # Phase 2: Train supervisor
            print("Phase 2: Training supervisor...")
            self._train_supervisor()
            
            # Phase 3: Joint Training
            print("Phase 3: Joint training of generator and discriminator...")
            self._joint_train()
            
            # Save trained models for this location
            self.location_models[location] = {
                'embedder': self.embedder,
                'recovery': self.recovery,
                'generator': self.generator,
                'discriminator': self.discriminator,
                'supervisor': self.supervisor,
                'synthetic_data': self.synthetic_data
            }
        
        return self
    
    def generate(self, n_samples, locations=None, time_index=None):
        """
        Generate synthetic interference data
        
        Args:
            n_samples: Number of synthetic sequences to generate per location
            locations: Locations to generate data for (if None, use all trained locations)
            time_index: Time index for the generated data
            
        Returns:
            DataFrame with synthetic interference data
        """
        if locations is None:
            locations = list(self.location_models.keys())
        
        all_generated_data = []
        
        for location in locations:
            if location not in self.location_models:
                print(f"Warning: No model found for location {location}. Skipping.")
                continue
                
            print(f"Generating {n_samples} synthetic interference sequences for {location}...")
            
            # Restore models for this location
            self.embedder = self.location_models[location]['embedder']
            self.recovery = self.location_models[location]['recovery']
            self.generator = self.location_models[location]['generator']
            self.discriminator = self.location_models[location]['discriminator']
            self.supervisor = self.location_models[location]['supervisor']
            self.synthetic_data = self.location_models[location]['synthetic_data']
            
            generated_data = []
            
            # Generate synthetic sequences
            for _ in range(int(np.ceil(n_samples / self.batch_size))):
                Z_ = next(self.random_series)
                d = self.synthetic_data(Z_)
                generated_data.append(d)
            
            # Stack and reshape
            generated_data = np.array(np.vstack(generated_data))
            
            # Inverse transform to get original scale
            scaler = self.location_scalers[location]
            generated_data = (scaler.inverse_transform(
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
            
            # Create DataFrame
            train_cols = [col for col in self.location_models.keys() 
                         if col not in ['Time', 'Location'] and 'interference' in col]
            
            if not train_cols:
                # If train_cols is empty, infer columns from existing data
                train_cols = [f"{band}_interference" for band in BANDS.keys()]
            
            df = pd.DataFrame(flat_data, columns=train_cols)
            df['Time'] = time_index[:len(df)]
            df['Location'] = location
            
            # Ensure valid values (0-100%)
            for col in train_cols:
                df[col] = df[col].clip(0, 100)
            
            all_generated_data.append(df)
        
        # Combine all locations
        return pd.concat(all_generated_data, ignore_index=True)

# Class for time-series analysis and correlation with demand data
class InterferenceDemandAnalyzer:
    """
    Analyze and correlate interference and demand data to extract features
    for training dynamic spectrum allocation models
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def load_data(self, demand_df, interference_df):
        """
        Load and merge demand and interference data
        
        Args:
            demand_df: DataFrame with spectrum demand data
            interference_df: DataFrame with interference data
            
        Returns:
            DataFrame with merged data
        """
        # Make copies to avoid modifying originals
        demand = demand_df.copy()
        interference = interference_df.copy()
        
        # Ensure Time is datetime
        demand['Time'] = pd.to_datetime(demand['Time'])
        interference['Time'] = pd.to_datetime(interference['Time'])
        
        # Group by time if multiple locations in interference data
        if 'Location' in interference.columns:
            # Aggregate interference across locations
            agg_dict = {col: 'mean' for col in interference.columns 
                       if 'interference' in col}
            interference_agg = interference.groupby('Time').agg(agg_dict).reset_index()
        else:
            interference_agg = interference
        
        # Merge datasets
        merged = pd.merge(demand, interference_agg, on='Time', how='inner')
        return merged
    
    def extract_features(self, merged_df):
        """
        Extract features for spectrum allocation model training
        
        Args:
            merged_df: DataFrame with merged demand and interference data
            
        Returns:
            DataFrame with features for model training
        """
        features = merged_df.copy()
        
        # Calculate signal-to-interference ratio for each band
        for band in BANDS.keys():
            # Ensure columns exist
            if band in features.columns and f"{band}_interference" in features.columns:
                # Signal-to-Interference Ratio (SIR)
                features[f"{band}_SIR"] = features[band] / (features[f"{band}_interference"] + 1)
                
                # Effective capacity (simplified model)
                features[f"{band}_effective_capacity"] = features[band] * (1 - (features[f"{band}_interference"] / 100))
        
        # Add time-based features
        features['hour'] = features['Time'].dt.hour
        features['day_of_week'] = features['Time'].dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_business_hours'] = ((features['hour'] >= 8) & (features['hour'] <= 18)).astype(int)
        
        # Rolling window features
        window_sizes = [3, 6, 12, 24]
        for window in window_sizes:
            for band in BANDS.keys():
                # Demand trends
                if band in features.columns:
                    features[f"{band}_demand_trend_{window}h"] = features[band].rolling(window).mean()
                
                # Interference trends
                if f"{band}_interference" in features.columns:
                    features[f"{band}_interference_trend_{window}h"] = features[f"{band}_interference"].rolling(window).mean()
        
        # Fill NaN values from rolling windows
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        return features
    
    def generate_allocation_scenarios(self, features_df, n_scenarios=5):
        """
        Generate spectrum allocation scenarios based on demand and interference
        
        Args:
            features_df: DataFrame with features
            n_scenarios: Number of scenarios to generate
            
        Returns:
            DataFrame with allocation scenarios
        """
        scenarios = []
        
        for i in range(n_scenarios):
            scenario = features_df.copy()
            
            # Scenario name
            scenario_name = f"scenario_{i+1}"
            scenario['scenario'] = scenario_name
            
            # Different allocation strategies
            if i == 0:
                # Baseline: Allocate based on pure demand
                for band in BANDS.keys():
                    if band in scenario.columns:
                        scenario[f"{band}_allocation"] = scenario[band] / 100
            
            elif i == 1:
                # Interference-aware: Allocate based on SIR
                for band in BANDS.keys():
                    if f"{band}_SIR" in scenario.columns:
                        scenario[f"{band}_allocation"] = (
                            scenario[f"{band}_SIR"] / scenario[[f"{b}_SIR" for b in BANDS.keys() 
                                                              if f"{b}_SIR" in scenario.columns]].sum(axis=1)
                        )
            
            elif i == 2:
                # Effective capacity maximization
                for band in BANDS.keys():
                    if f"{band}_effective_capacity" in scenario.columns:
                        scenario[f"{band}_allocation"] = (
                            scenario[f"{band}_effective_capacity"] / 
                            scenario[[f"{b}_effective_capacity" for b in BANDS.keys() 
                                     if f"{b}_effective_capacity" in scenario.columns]].sum(axis=1)
                        )
            
            elif i == 3:
                # Time-aware allocation (different strategies for different times)
                for band in BANDS.keys():
                    # Use effective capacity as base
                    if f"{band}_effective_capacity" in scenario.columns:
                        base_allocation = scenario[f"{band}_effective_capacity"]
                        
                        # Boost low bands during busy hours for reliability
                        if 'Low' in band:
                            boost = scenario['is_business_hours'] * 0.2 + 1.0
                            base_allocation = base_allocation * boost
                        
                        # Boost high bands during night for high bandwidth applications
                        if 'High' in band:
                            boost = (1 - scenario['is_business_hours']) * 0.3 + 1.0
                            base_allocation = base_allocation * boost
                        
                        scenario[f"{band}_allocation"] = base_allocation
                
                # Normalize allocations to sum to 1
                allocation_cols = [f"{band}_allocation" for band in BANDS.keys() 
                                  if f"{band}_allocation" in scenario.columns]
                allocation_sum = scenario[allocation_cols].sum(axis=1)
                for col in allocation_cols:
                    scenario[col] = scenario[col] / allocation_sum
            
            else:
                # Adaptive learning-based approach (simulated)
                for band in BANDS.keys():
                    if band in scenario.columns and f"{band}_interference" in scenario.columns:
                        # Complex allocation based on multiple factors
                        demand_weight = 0.5
                        interference_weight = 0.3
                        time_weight = 0.2
                        
                        # Normalized demand contribution
                        demand_factor = scenario[band] / 100
                        
                        # Inverse interference contribution (lower is better)
                        interference_factor = 1 - (scenario[f"{band}_interference"] / 100)
                        
                        # Time-based factor
                        if 'Low' in band:
                            # Low bands preferred during business hours
                            time_factor = scenario['is_business_hours'] * 0.2 + 0.9
                        elif 'Mid' in band:
                            # Mid bands always steady
                            time_factor = np.ones(len(scenario))
                        else:  # High
                            # High bands preferred during non-business hours
                            time_factor = (1 - scenario['is_business_hours']) * 0.3 + 0.8
                        
                        # Combined allocation score
                        scenario[f"{band}_allocation"] = (
                            demand_weight * demand_factor +
                            interference_weight * interference_factor +
                            time_weight * time_factor
                        )
                
                # Normalize allocations to sum to 1
                allocation_cols = [f"{band}_allocation" for band in BANDS.keys() 
                                  if f"{band}_allocation" in scenario.columns]
                allocation_sum = scenario[allocation_cols].sum(axis=1)
                for col in allocation_cols:
                    scenario[col] = scenario[col] / allocation_sum
            
            scenarios.append(scenario)
        
        # Combine all scenarios
        return pd.concat(scenarios, ignore_index=True)

# Main execution function for interference data generation
def generate_spectrum_interference_dataset(sample_size=30, seq_length=24, num_iterations=5, 
                                         output_dir="spectrum_interference_data",
                                         locations=None):
    """
    Generate synthetic spectrum interference data across different frequency bands and locations
    
    Args:
        sample_size: Number of days to generate in each iteration
        seq_length: Sequence length for TimeGAN
        num_iterations: Number of different datasets to generate
        output_dir: Directory to save output files
        locations: List of locations to include (if None, use all LOCATIONS)
    """
    print(f"Generating synthetic spectrum interference data with TimeGAN")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use specified locations or default list
    if locations is None:
        locations = LOCATIONS
    
    # Generate initial sample data
    print("Generating initial sample interference data...")
    sample_df = generate_interference_sample_data(days=sample_size, locations=locations)
    
    # Save sample data
    sample_df.to_csv(os.path.join(output_dir, "sample_interference_data.csv"), index=False)
    
    # Set up TimeGAN model
    model = InterferenceTimeGAN(
        seq_len=seq_length,
        hidden_dim=32,
        num_layers=3,
        batch_size=64,
        train_steps=5000  # Use higher value for production
    )
    
    # Train the model
    print("Training TimeGAN model on interference data...")
    model.fit(sample_df, locations=locations)
    
    # Generate synthetic datasets
    for i in range(num_iterations):
        print(f"Generating synthetic dataset {i+1}/{num_iterations}")
        
        # Generate synthetic data (approximately 3x the original size per location)
        synth_df = model.generate(n_samples=sample_size*3//len(locations), locations=locations)
        
        # Save the dataset
        output_file = os.path.join(output_dir, f"spectrum_interference_{i+1}.csv")
        synth_df.to_csv(output_file, index=False)
        print(f"Saved synthetic dataset to {output_file}")
        
        # Create and save visualizations for first iteration
        if i == 0:
            visualize_interference_data(sample_df, synth_df, output_dir)
    
    print(f"Successfully generated {num_iterations} synthetic spectrum interference datasets in {output_dir}")
    return True

def visualize_interference_data(real_df, synth_df, output_dir):
    """Create visualizations comparing real and synthetic interference data"""
    # Create a subdirectory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get all locations
    locations = real_df['Location'].unique()
    interference_cols = [col for col in real_df.columns if 'interference' in col]
    
    # Generate visualizations for each location
    for location in locations:
        # Create location-specific directory
        loc_dir = os.path.join(viz_dir, location.replace(" ", "_"))
        os.makedirs(loc_dir, exist_ok=True)
        
        # Select data for this location
        real_subset = real_df[real_df['Location'] == location].iloc[:168]  # One week
        synth_subset = synth_df[synth_df['Location'] == location].iloc[:168]  # One week
        
        # Band comparison plot
        plt.figure(figsize=(15, 12))
        n_cols = len(interference_cols)
        for i, col in enumerate(interference_cols):
            plt.subplot(n_cols, 1, i+1)
            plt.plot(real_subset['Time'], real_subset[col], 'b-', label='Original', alpha=0.7)
            plt.plot(synth_subset['Time'], synth_subset[col], 'r-', label='Synthetic', alpha=0.7)
            plt.title(f'{col} - Location: {location}')
            plt.ylabel('Interference Level (%)')
            plt.legend()
            if i == n_cols - 1:
                plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, 'interference_comparison.png'))
        plt.close()
        
        # Correlation heatmaps
        plt.figure(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        corr_real = real_df[real_df['Location'] == location][interference_cols].corr()
        sns.heatmap(corr_real, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Original Interference Correlation - {location}')
        
        plt.subplot(1, 2, 2)
        corr_synth = synth_df[synth_df['Location'] == location][interference_cols].corr()
        sns.heatmap(corr_synth, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Synthetic Interference Correlation - {location}')
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, 'interference_correlation.png'))
        plt.close()
        
        # Distribution comparison
        plt.figure(figsize=(15, 12))
        rows = (len(interference_cols) + 1) // 2
        for i, col in enumerate(interference_cols):
            plt.subplot(rows, 2, i+1)
            sns.histplot(real_subset[col], kde=True, color='blue', alpha=0.5, label='Original')
            sns.histplot(synth_subset[col], kde=True, color='red', alpha=0.5, label='Synthetic')
            plt.title(f'{col} Distribution - {location}')
            plt.xlabel('Interference Level (%)')
            plt.ylabel('Frequency')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, 'interference_distribution.png'))
        plt.close()
        
        # Time-based patterns
        plt.figure(figsize=(15, 10))
        example_band = interference_cols[0]
        
        # Hourly patterns
        plt.subplot(2, 1, 1)
        hourly_real = real_df[real_df['Location'] == location].groupby(real_df['Time'].dt.hour)[example_band].mean()
        hourly_synth = synth_df[synth_df['Location'] == location].groupby(synth_df['Time'].dt.hour)[example_band].mean()
        
        plt.plot(hourly_real.index, hourly_real.values, 'b-', label='Original', linewidth=2)
        plt.plot(hourly_synth.index, hourly_synth.values, 'r-', label='Synthetic', linewidth=2)
        plt.title(f'Hourly Pattern: {example_band} - {location}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Avg Interference Level (%)')
        plt.xticks(range(0, 24, 2))
        plt.legend()
        
        # Day of week patterns
        plt.subplot(2, 1, 2)
        day_real = real_df[real_df['Location'] == location].groupby(real_df['Time'].dt.dayofweek)[example_band].mean()
        day_synth = synth_df[synth_df['Location'] == location].groupby(synth_df['Time'].dt.dayofweek)[example_band].mean()
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.plot(day_real.index, day_real.values, 'b-', label='Original', linewidth=2)
        plt.plot(day_synth.index, day_synth.values, 'r-', label='Synthetic', linewidth=2)
        plt.title(f'Day of Week Pattern: {example_band} - {location}')
        plt.xlabel('Day of Week')
        plt.ylabel('Avg Interference Level (%)')
        plt.xticks(range(7), days, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(loc_dir, 'interference_time_patterns.png'))
        plt.close()

# Adding the missing SpectrumTimeGAN class that InterferenceTimeGAN extends
class SpectrumTimeGAN:
    """
    Implementation of TimeGAN (Time-series Generative Adversarial Network) 
    for spectrum demand and interference data.
    
    Based on "Time-series Generative Adversarial Networks" by Yoon et al.
    """
    
    def __init__(self, seq_len=24, hidden_dim=24, num_layers=3, 
                 batch_size=128, train_steps=2000, learning_rate=0.001):
        """
        Initialize the TimeGAN model
        
        Args:
            seq_len: Length of the sequence to generate
            hidden_dim: Dimension of hidden layers
            num_layers: Number of layers in RNN networks
            batch_size: Batch size for training
            train_steps: Number of steps for each training phase
            learning_rate: Learning rate for Adam optimizer
        """
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        
        # Will be determined when fitting the model
        self.n_seq = None  # Number of features in the time series
        
    def _build_networks(self):
        """Build the component networks of TimeGAN"""
        # Embedder: time-series -> latent representation
        embedder_input = Input(shape=(self.seq_len, self.n_seq))
        embedder_hidden = GRU(units=self.hidden_dim,
                             return_sequences=True,
                             name='embedder_gru')(embedder_input)
        embedder_output = Dense(units=self.hidden_dim,
                             activation='sigmoid',
                             name='embedder_dense')(embedder_hidden)
        self.embedder = Model(embedder_input, embedder_output, name='embedder')
        
        # Recovery: latent representation -> time-series
        recovery_input = Input(shape=(self.seq_len, self.hidden_dim))
        recovery_hidden = GRU(units=self.hidden_dim,
                            return_sequences=True,
                            name='recovery_gru')(recovery_input)
        recovery_output = Dense(units=self.n_seq,
                              activation='sigmoid',
                              name='recovery_dense')(recovery_hidden)
        self.recovery = Model(recovery_input, recovery_output, name='recovery')
        
        # Generator: random noise -> latent representation
        generator_input = Input(shape=(self.seq_len, self.hidden_dim))
        generator_hidden = GRU(units=self.hidden_dim,
                             return_sequences=True,
                             name='generator_gru')(generator_input)
        generator_output = Dense(units=self.hidden_dim,
                               activation='sigmoid',
                               name='generator_dense')(generator_hidden)
        self.generator = Model(generator_input, generator_output, name='generator')
        
        # Discriminator: time-series or latent -> binary classification
        discriminator_input = Input(shape=(self.seq_len, self.hidden_dim))
        discriminator_hidden = GRU(units=self.hidden_dim,
                                 return_sequences=False,
                                 name='discriminator_gru')(discriminator_input)
        discriminator_output = Dense(units=1,
                                   activation='sigmoid',
                                   name='discriminator_dense')(discriminator_hidden)
        self.discriminator = Model(discriminator_input, discriminator_output, name='discriminator')
        
        # Supervisor: latent -> latent for stepwise supervision
        supervisor_input = Input(shape=(self.seq_len, self.hidden_dim))
        supervisor_hidden = GRU(units=self.hidden_dim,
                              return_sequences=True,
                              name='supervisor_gru')(supervisor_input)
        supervisor_output = Dense(units=self.hidden_dim,
                                activation='sigmoid',
                                name='supervisor_dense')(supervisor_hidden)
        self.supervisor = Model(supervisor_input, supervisor_output, name='supervisor')
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        
        # Define synthetic data generator (G + R)
        random_input = Input(shape=(self.seq_len, self.hidden_dim))
        generator_output = self.generator(random_input)
        synthetic_output = self.recovery(generator_output)
        self.synthetic_data = Model(random_input, synthetic_output, name='synthetic_data')
        
    def _train_autoencoder(self):
        """Train the embedder and recovery networks"""
        # Define autoencoder model (E + R)
        input_data = Input(shape=(self.seq_len, self.n_seq))
        encoder_output = self.embedder(input_data)
        decoder_output = self.recovery(encoder_output)
        autoencoder = Model(input_data, decoder_output)
        
        # Compile autoencoder
        autoencoder.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        # Train autoencoder
        for step in range(self.train_steps):
            X = next(self.real_series_iter)
            loss = autoencoder.train_on_batch(X, X)
            
            if step % 100 == 0:
                print(f'Autoencoder Loss: {loss:.4f}')
    
    def _train_supervisor(self):
        """Train the supervisor network"""
        # Define supervised model
        input_data = Input(shape=(self.seq_len, self.n_seq))
        embed_data = self.embedder(input_data)
        supervisor_output = self.supervisor(embed_data)
        supervised_model = Model(input_data, supervisor_output)
        
        # Compile supervised model
        supervised_model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        # Train supervisor
        for step in range(self.train_steps):
            X = next(self.real_series_iter)
            # Get the embedding representations
            H = self.embedder.predict(X, verbose=0)
            # Let the supervisor predict the next sequence for training
            # Use H[:,:-1,:] as input and H[:,1:,:] as target
            loss = supervised_model.train_on_batch(X, H)
            
            if step % 100 == 0:
                print(f'Supervisor Loss: {loss:.4f}')
    
    def _joint_train(self):
        """Train the whole network jointly"""
        # Freeze discriminator during generator training
        self.discriminator.trainable = False
        
        # Joint model for adversarial training
        # Generator
        generator_input = Input(shape=(self.seq_len, self.hidden_dim))
        generator_output = self.generator(generator_input)
        supervisor_output = self.supervisor(generator_output)
        discriminator_output = self.discriminator(generator_output)
        synthetic_output = self.recovery(generator_output)
        
        # Joint model
        joint_model = Model(
            generator_input, 
            [discriminator_output, synthetic_output, supervisor_output]
        )
        
        # Compile joint model
        joint_model.compile(
            loss=['binary_crossentropy', 'mse', 'mse'],
            loss_weights=[1.0, 1.0, 1.0],
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        # Train the discriminator and generator jointly
        for step in range(self.train_steps):
            # Discriminator training
            self.discriminator.trainable = True
            
            # Real data
            X_real = next(self.real_series_iter)
            H_real = self.embedder.predict(X_real, verbose=0)
            Y_real = np.ones((len(X_real), 1))
            
            # Synthetic data
            Z = next(self.random_series)
            H_fake = self.generator.predict(Z, verbose=0)
            Y_fake = np.zeros((len(Z), 1))
            
            # Train discriminator
            D_loss_real = self.discriminator.train_on_batch(H_real, Y_real)
            D_loss_fake = self.discriminator.train_on_batch(H_fake, Y_fake)
            D_loss = 0.5 * (D_loss_real[0] + D_loss_fake[0])
            D_accuracy = 0.5 * (D_loss_real[1] + D_loss_fake[1])
            
            # Generator training
            self.discriminator.trainable = False
            
            # Train generator
            Z = next(self.random_series)
            Y_fake = np.ones((len(Z), 1))  # Try to fool discriminator
            
            # Sample from real data for supervised loss
            X_real = next(self.real_series_iter)
            H_real = self.embedder.predict(X_real, verbose=0)
            
            # Train the joint model
            G_loss = joint_model.train_on_batch(
                Z,
                [Y_fake, X_real, H_real]
            )
            
            if step % 100 == 0:
                print(f'Step: {step}')
                print(f'D Loss: {D_loss:.4f}, D Accuracy: {D_accuracy:.4f}')
                print(f'G Loss: {G_loss[0]:.4f}')
                print(f'G Components: Adversarial: {G_loss[1]:.4f}, Reconstruction: {G_loss[2]:.4f}, Supervised: {G_loss[3]:.4f}')
                print('-' * 50)

# Define a function to train and evaluate allocation models
def train_dynamic_spectrum_allocation_models(features_df, scenario_df, output_dir):
    """
    Train models for dynamic spectrum allocation based on demand and interference
    
    Args:
        features_df: DataFrame with features from InterferenceDemandAnalyzer
        scenario_df: DataFrame with allocation scenarios
        output_dir: Directory to save model outputs
    
    Returns:
        Dictionary with trained models
    """
    print("Training dynamic spectrum allocation models...")
    
    # Create output directory for models
    model_dir = os.path.join(output_dir, "allocation_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Get available bands
    bands = [band for band in BANDS.keys() if f"{band}_allocation" in scenario_df.columns]
    
    # Prepare training data
    allocation_targets = [f"{band}_allocation" for band in bands]
    
    # Select features (exclude allocation targets and scenario column)
    feature_cols = [col for col in features_df.columns 
                   if col not in allocation_targets + ['Time', 'scenario'] 
                   and not pd.isna(col)]
    
    # Extract X and y
    X = features_df[feature_cols].values
    y = scenario_df[allocation_targets].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Build neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(bands), activation='softmax')  # Softmax ensures allocations sum to 1
    ])
    
    # Compile model
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001),
        metrics=['mae']
    )
    
    # Train model
    print("Training neural network model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Save model
    model.save(os.path.join(model_dir, "spectrum_allocation_model.h5"))
    
    # Save feature scaler
    import joblib
    joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    
    # Visualize training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_history.png"))
    
    # Visualize predicted vs actual allocations
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(15, 10))
    
    for i, band in enumerate(bands):
        plt.subplot(len(bands), 1, i+1)
        plt.plot(y_test[:100, i], 'b-', label=f'Actual {band}')
        plt.plot(y_pred[:100, i], 'r-', label=f'Predicted {band}')
        plt.title(f'Spectrum Allocation for {band}')
        plt.xlabel('Time Step')
        plt.ylabel('Allocation Ratio')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "allocation_predictions.png"))
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "bands": bands
    }

# Full pipeline function to run the entire process
def run_spectrum_management_pipeline(output_dir="spectrum_management_data", days=30, iterations=3):
    """
    Run the pipeline to generate interference data:
    1. Generate demand data
    2. Generate interference data
    
    Args:
        output_dir: Directory to save all outputs
        days: Number of days to simulate
        iterations: Number of different datasets to generate
    """

    try:
        print("Starting Interference Data Generation Pipeline")
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Generate demand data
        demand_dir = os.path.join(output_dir, "demand_data")
        os.makedirs(demand_dir, exist_ok=True)
        
        # Generate sample demand data directly
        print("Generating sample demand data...")
        demand_sample = pd.DataFrame()
        
        # Create time series
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(hours=h) for h in range(24 * days)]
        demand_sample['Time'] = dates
        
        # Generate synthetic demand for each band
        for band in BANDS.keys():
            # Base demand pattern
            base_demand = np.random.normal(50, 15, len(dates))
            
            # Add time-of-day variation
            hours = pd.to_datetime(dates).hour
            time_factor = 20 * np.sin(2 * np.pi * (hours - 12) / 24)
            
            # Add day-of-week variation
            days_of_week = pd.to_datetime(dates).dayofweek
            weekend_factor = 10 * (days_of_week >= 5)
            
            # Combine factors
            demand = base_demand + time_factor + weekend_factor
            
            # Ensure values are between 0 and 100
            demand = np.clip(demand, 0, 100)
            
            demand_sample[band] = demand
        
        # Save sample demand data
        demand_sample.to_csv(os.path.join(demand_dir, "sample_demand_data.csv"), index=False)
        
        # 2. Generate interference data
        interference_dir = os.path.join(output_dir, "interference_data")
        print("Generating spectrum interference data...")
        generate_spectrum_interference_dataset(
            sample_size=days,
            seq_length=24,
            num_iterations=iterations,
            output_dir=interference_dir
        )
        
        print("Interference Data Generation Pipeline completed successfully")
        return True

    finally:
        # Cleanup TensorFlow resources
        import tensorflow as tf
        tf.keras.backend.clear_session()

# If this module is run directly, execute the pipeline
if __name__ == "__main__":
    run_spectrum_management_pipeline()