import os
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_generator import ReportGenerator, REPORTS_DIR

def main():
    print("Starting time series report generation...")
    
    # Create report generator
    generator = ReportGenerator()
    
    # Generate sample data if needed
    spectrum_data_path = os.path.join(os.path.dirname(__file__), "outputs", "cnn_future_predictions.csv")
    interference_data_path = os.path.join(os.path.dirname(__file__), "outputs", "recent_interference_data.csv")
    
    # Create sample spectrum data if it doesn't exist
    if not os.path.exists(spectrum_data_path):
        print(f"Creating sample spectrum data at {spectrum_data_path}")
        create_sample_spectrum_data(spectrum_data_path)
    
    # Create sample interference data if it doesn't exist
    if not os.path.exists(interference_data_path):
        print(f"Creating sample interference data at {interference_data_path}")
        create_sample_interference_data(interference_data_path)
    
    # Create a directory for time series reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_series_dir = os.path.join(REPORTS_DIR, f"time_series_{timestamp}")
    os.makedirs(time_series_dir, exist_ok=True)
    print(f"Created directory for time series reports: {time_series_dir}")
    
    # Load data
    try:
        spectrum_df = pd.read_csv(spectrum_data_path)
        interference_df = pd.read_csv(interference_data_path)
        
        print(f"Loaded spectrum data with {len(spectrum_df)} rows")
        print(f"Loaded interference data with {len(interference_df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate a report for each time step
    base_data = generator.collect_data()
    
    # Use the smaller dataset as our base
    num_reports = min(len(spectrum_df), len(interference_df), 48)  # Limit to 48 reports max
    print(f"Generating {num_reports} time series reports...")
    
    for i in range(num_reports):
        # Create a unique timestamp for this report
        current_time = datetime.now() - timedelta(hours=i)
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create data for this specific time step
        time_data = base_data.copy()
        time_data['timestamp'] = time_str
        
        # Get spectrum data for this time step
        if i < len(spectrum_df):
            spectrum_row = spectrum_df.iloc[i]
            
            # Extract demand columns
            demand_cols = [col for col in spectrum_df.columns if 'Demand' in col]
            
            if demand_cols:
                # Calculate metrics with random variations
                avg_demand = spectrum_row[demand_cols].mean() if isinstance(spectrum_row[demand_cols].mean(), (int, float)) else random.uniform(0.3, 0.8)
                demand_std = spectrum_row[demand_cols].std() if isinstance(spectrum_row[demand_cols].std(), (int, float)) else random.uniform(0.05, 0.2)
                
                # Add variations to make each report unique
                avg_demand = avg_demand * random.uniform(0.9, 1.1)
                demand_std = demand_std * random.uniform(0.9, 1.1)
                
                # Calculate efficiency score
                efficiency_score = 1.0 - (demand_std / (avg_demand + 1e-5))
                efficiency_score = max(0.5, min(0.95, efficiency_score))
                
                # Determine congestion level
                congestion_options = ["low", "medium", "high"]
                congestion_weights = [0.2, 0.3, 0.5] if avg_demand > 0.7 else \
                                    [0.3, 0.5, 0.2] if avg_demand > 0.4 else \
                                    [0.6, 0.3, 0.1]
                congestion_level = random.choices(congestion_options, weights=congestion_weights)[0]
                
                # Update spectrum allocation data
                time_data["spectrum_allocation"] = {
                    "channels_allocated": len(demand_cols) + random.randint(-1, 1),
                    "efficiency_score": round(float(efficiency_score), 2),
                    "congestion_level": congestion_level,
                    "prediction_horizon": 24,
                    "average_demand": round(float(avg_demand), 2),
                    "demand_std": round(float(demand_std), 2)
                }
        
        # Get interference data for this time step
        if i < len(interference_df):
            interference_row = interference_df.iloc[i]
            
            # Extract interference columns
            interference_cols = [col for col in interference_df.columns if 'interference' in col.lower()]
            
            if interference_cols:
                # Get base values
                g4_cols = [col for col in interference_cols if '4G' in col]
                g5_cols = [col for col in interference_cols if '5G' in col]
                
                g4_value = interference_row[g4_cols].mean() if g4_cols and isinstance(interference_row[g4_cols].mean(), (int, float)) else random.uniform(0.2, 0.6)
                g5_value = interference_row[g5_cols].mean() if g5_cols and isinstance(interference_row[g5_cols].mean(), (int, float)) else random.uniform(0.2, 0.6)
                
                # Add variations
                g4_value = g4_value * random.uniform(0.85, 1.15)
                g5_value = g5_value * random.uniform(0.85, 1.15)
                
                # Determine interference types
                interference_types = []
                severity_levels = {}
                
                # Vary thresholds
                g4_threshold = random.uniform(0.35, 0.45)
                g5_threshold = random.uniform(0.35, 0.45)
                
                if g4_value > g4_threshold:
                    interference_types.append("co_channel")
                    severity = "high" if g4_value > 0.6 else "medium" if g4_value > 0.4 else "low"
                    severity_levels["co_channel"] = severity
                
                if g5_value > g5_threshold:
                    interference_types.append("wideband")
                    severity = "high" if g5_value > 0.6 else "medium" if g5_value > 0.4 else "low"
                    severity_levels["wideband"] = severity
                
                # Sometimes add adjacent_channel
                if random.random() < 0.4 or not interference_types:
                    interference_types.append("adjacent_channel")
                    severity_options = ["low", "medium", "high"]
                    weights = [0.6, 0.3, 0.1]
                    severity_levels["adjacent_channel"] = random.choices(severity_options, weights=weights)[0]
                
                # Determine affected bands
                affected_bands = []
                if g4_value > 0.3 or random.random() < 0.3:
                    affected_bands.append("2.4GHz")
                if g5_value > 0.3 or random.random() < 0.4:
                    affected_bands.append("5GHz")
                if random.random() < 0.2:
                    affected_bands.append("6GHz")
                
                # Ensure at least one band
                if not affected_bands:
                    affected_bands = ["2.4GHz"]
                
                # Update interference classification
                time_data["interference_classification"] = {
                    "interference_types": interference_types,
                    "severity_levels": severity_levels,
                    "affected_bands": affected_bands
                }
                
                # Add location if available
                if 'Location' in interference_row:
                    time_data["interference_classification"]["location"] = interference_row['Location']
                else:
                    # Generate random location
                    locations = ["urban", "suburban", "rural", "industrial"]
                    time_data["interference_classification"]["location"] = random.choice(locations)
        
        # Generate report for this time step
        report_filename = f"spectrum_report_{current_time.strftime('%Y%m%d_%H%M')}.txt"
        report_path = os.path.join(time_series_dir, report_filename)
        
        # Generate report with template
        report = generator._generate_template_report(time_data)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        if i % 5 == 0 or i == num_reports - 1:
            print(f"Generated report {i+1}/{num_reports}: {report_filename}")
    
    print(f"Generated {num_reports} time series reports in {time_series_dir}")

def create_sample_spectrum_data(output_path):
    """Create sample spectrum data file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sample data for 48 hours
    now = datetime.now()
    times = [(now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(48)]
    
    # Create DataFrame with demand columns
    df = pd.DataFrame({
        'Time': times,
        'Demand_Channel1': np.random.uniform(0.3, 0.8, 48),
        'Demand_Channel2': np.random.uniform(0.2, 0.7, 48),
        'Demand_Channel3': np.random.uniform(0.4, 0.9, 48),
        'Demand_Channel4': np.random.uniform(0.1, 0.6, 48),
        'Demand_Channel5': np.random.uniform(0.2, 0.5, 48),
        'Demand_Channel6': np.random.uniform(0.3, 0.7, 48)
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created sample spectrum data at {output_path}")

def create_sample_interference_data(output_path):
    """Create sample interference data file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sample data for 48 hours
    now = datetime.now()
    times = [(now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(48)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': times,
        '4G_Low_interference': np.random.uniform(0.1, 0.5, 48),
        '4G_Mid_interference': np.random.uniform(0.2, 0.6, 48),
        '4G_High_interference': np.random.uniform(0.1, 0.4, 48),
        '5G_Low_interference': np.random.uniform(0.3, 0.7, 48),
        '5G_Mid_interference': np.random.uniform(0.2, 0.5, 48),
        '5G_High_interference': np.random.uniform(0.1, 0.3, 48),
        'Location': np.random.choice(['urban', 'suburban', 'rural', 'industrial'], 48)
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created sample interference data at {output_path}")

if __name__ == "__main__":
    main()