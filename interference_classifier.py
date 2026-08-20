import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Define area priority levels
AREA_PRIORITY = {
    'urban_dense': 3,      # Highest priority
    'downtown': 3,
    'shopping_mall': 3,
    'campus': 2,
    'urban_sparse': 2,
    'industrial_zone': 2,
    'industrial': 2,
    'airport': 2,
    'seaport': 2,
    'suburban': 1,
    'highway': 1,
    'rural': 0             # Lowest priority
}

def load_and_prepare_data(file_path):
    """Load and prepare data with area-based features"""
    df = pd.read_csv(file_path)
    
    # Prepare features
    feature_cols = []
    bands = ['4G_Low', '4G_Mid', '4G_High', '5G_Low', '5G_Mid', '5G_High']
    for band in bands:
        feature_cols.append(f"{band}_interference")
    
    # Create time-based features
    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    
    feature_cols.extend(['hour', 'day_of_week', 'is_weekend', 'is_business_hours'])
    
    # Add area priority as a feature
    if 'Location' in df.columns:
        df['area_priority'] = df['Location'].map(AREA_PRIORITY).fillna(0)
        feature_cols.append('area_priority')
        
        # Create area-based target variable
        return df[feature_cols].values, df['Location'].values
    
    return df[feature_cols].values, None

def create_sequences(data, sequence_length=24):
    """Create sequences for CNN classification"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def build_cnn_model(input_shape, num_classes):
    """Build CNN model for area-based interference classification"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC()]
    )
    
    return model

def calculate_priority_score(interference_data, area_labels, confidence_scores):
    """Calculate priority scores for spectrum allocation"""
    # Extract interference levels (first 6 columns)
    interference_levels = np.mean(interference_data[:, :6], axis=1)
    
    # Ensure all arrays have the same length
    if len(interference_levels) != len(area_labels) or len(interference_levels) != len(confidence_scores):
        # Reshape or truncate arrays to match
        min_length = min(len(interference_levels), len(area_labels), len(confidence_scores))
        interference_levels = interference_levels[:min_length]
        area_labels = area_labels[:min_length]
        confidence_scores = confidence_scores[:min_length]
        
        print(f"Warning: Arrays had different lengths. Truncated to {min_length} elements.")
    
    # Get area priority values
    area_priority_values = np.array([AREA_PRIORITY.get(area, 0) for area in area_labels])
    
    # Calculate priority score: interference level * area priority * prediction confidence
    priority_scores = interference_levels * (area_priority_values + 1) * confidence_scores
    
    return priority_scores

def train_model(data_file, output_dir, sequence_length=24, epochs=50):
    """Train CNN model for area-based interference classification"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data, area_labels = load_and_prepare_data(data_file)
    
    if area_labels is None:
        print("Error: Location data not found in the dataset")
        return None
    
    # Encode area labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(area_labels)
    
    # Save label encoder classes for later use
    np.save(os.path.join(output_dir, 'label_encoder_classes.npy'), label_encoder.classes_)
    
    # Create sequences
    print("Creating sequences...")
    sequences = create_sequences(data, sequence_length)
    
    # Adjust labels to match sequences
    labels = encoded_labels[sequence_length-1:]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Convert labels to categorical
    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Build model
    print("Building model...")
    model = build_cnn_model((sequence_length, data.shape[1]), num_classes)
    model.summary()
    
    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Changed file extension from .h5 to .keras
    model_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'interference_classifier_cnn.keras'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_cat),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = model.evaluate(X_test_scaled, y_test_cat)
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}")
    print(f"Test AUC: {evaluation[2]:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_cat, axis=1)
    
    # Get confidence scores
    confidence_scores = np.max(predictions, axis=1)
    
    # Get area labels
    predicted_areas = label_encoder.inverse_transform(predicted_classes)
    true_areas = label_encoder.inverse_transform(true_classes)
    
    # Calculate priority scores
    test_features = X_test[:, -1, :]  # Take the last timestep of each sequence
    priority_scores = calculate_priority_score(
        test_features, predicted_areas, confidence_scores
    )
    
    # Identify high priority areas
    high_priority_threshold = np.percentile(priority_scores, 75)  # Top 25%
    high_priority_indices = np.where(priority_scores >= high_priority_threshold)[0]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, 
                                  target_names=label_encoder.classes_,
                                  output_dict=False)
    print(report)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Print confusion matrix with area labels
    print("Rows: True Areas, Columns: Predicted Areas")
    area_labels = label_encoder.classes_
    # Print column headers
    print("      " + " ".join([f"{area[:8]:10s}" for area in area_labels]))
    for i, row in enumerate(cm):
        print(f"{area_labels[i][:8]:8s} {' '.join([f'{x:10d}' for x in row])}")

    # Print high priority areas
    print("\nHigh Priority Areas for Spectrum Allocation:")
    high_priority_areas = predicted_areas[high_priority_indices]
    unique_areas, area_counts = np.unique(high_priority_areas, return_counts=True)
    
    print(f"Number of high priority instances: {len(high_priority_indices)}")
    print("\nArea Distribution in High Priority Segments:")
    for area, count in zip(unique_areas, area_counts):
        percentage = (count / len(high_priority_areas)) * 100
        print(f"{area}: {count} instances ({percentage:.1f}%)")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Create improved priority visualization by area
    plt.figure(figsize=(12, 8))
    
    # Calculate average priority score by area
    area_priority_avg = {}
    for area in np.unique(predicted_areas):
        mask = predicted_areas == area
        area_priority_avg[area] = np.mean(priority_scores[mask])
    
    # Sort areas by priority score
    sorted_areas = sorted(area_priority_avg.items(), key=lambda x: x[1], reverse=True)
    areas = [a[0] for a in sorted_areas]
    scores = [a[1] for a in sorted_areas]
    
    # Create bar chart
    bars = plt.bar(areas, scores, color='lightblue')
    
    # Highlight high priority areas
    high_priority_areas = set(predicted_areas[high_priority_indices])
    for i, area in enumerate(areas):
        if area in high_priority_areas:
            bars[i].set_color('darkred')
    
    plt.axhline(high_priority_threshold, color='r', linestyle='--', label='High Priority Threshold')
    plt.title('Average Priority Score by Area')
    plt.xlabel('Area')
    plt.ylabel('Average Priority Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(['High Priority Threshold', 'Regular Areas', 'High Priority Areas'])
    plt.savefig(os.path.join(output_dir, 'area_priority_scores.png'))
    
    return model

def test_model(model_dir, test_file):
    """Test the trained model on new data"""
    # Load model and label encoder classes
    model = load_model(os.path.join(model_dir, 'interference_classifier_cnn.keras'))
    # Add allow_pickle=True to fix the error
    label_encoder_classes = np.load(os.path.join(model_dir, 'label_encoder_classes.npy'), allow_pickle=True)
    
    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data, true_labels = load_and_prepare_data(test_file)
    
    # Create sequences
    sequence_length = 24
    test_sequences = create_sequences(test_data, sequence_length)
    
    # Scale the data
    scaler = StandardScaler()
    test_reshaped = test_sequences.reshape(-1, test_sequences.shape[-1])
    test_scaled = scaler.fit_transform(test_reshaped)
    test_scaled = test_scaled.reshape(test_sequences.shape)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get predicted labels
    predicted_labels = label_encoder_classes[predicted_classes]
    
    # Get confidence scores
    confidence_scores = np.max(predictions, axis=1)
    avg_confidence = np.mean(confidence_scores)
    print(f"\nAverage prediction confidence: {avg_confidence:.2%}")
    
    # Calculate priority scores
    test_features = test_sequences[:, -1, :]  # Take the last timestep of each sequence
    priority_scores = calculate_priority_score(
        test_features, predicted_labels, confidence_scores
    )
    
    # Identify high priority areas
    high_priority_threshold = np.percentile(priority_scores, 75)  # Top 25%
    high_priority_indices = np.where(priority_scores >= high_priority_threshold)[0]
    
    # Print distribution of classifications
    print("\nDistribution of Classifications:")
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(predicted_labels)) * 100
        print(f"{label}: {count} instances ({percentage:.1f}%)")
    
    # Print high priority areas
    print("\nHigh Priority Areas for Spectrum Allocation:")
    high_priority_areas = predicted_labels[high_priority_indices]
    unique_areas, area_counts = np.unique(high_priority_areas, return_counts=True)
    
    print(f"Number of high priority instances: {len(high_priority_indices)}")
    print("\nArea Distribution in High Priority Segments:")
    for area, count in zip(unique_areas, area_counts):
        percentage = (count / len(high_priority_areas)) * 100
        print(f"{area}: {count} instances ({percentage:.1f}%)")
    
    # Evaluate if true labels are available
    if true_labels is not None:
        # Adjust true labels to match the sequence predictions
        true_labels = true_labels[sequence_length-1:]
        
        # Encode true labels
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        
        # Handle potential unknown labels
        valid_mask = np.isin(true_labels, label_encoder_classes)
        if not np.all(valid_mask):
            print(f"\nWarning: {np.sum(~valid_mask)} unknown labels found in test data")
            true_labels = true_labels[valid_mask]
            predicted_labels = predicted_labels[valid_mask]
        
        true_classes = label_encoder.transform(true_labels)
        predicted_classes = label_encoder.transform(predicted_labels)
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Print confusion matrix with area labels
        print("Rows: True Areas, Columns: Predicted Areas")
        area_labels = label_encoder_classes
        # Print column headers
        print("      " + " ".join([f"{area[:8]:10s}" for area in area_labels]))
        for i, row in enumerate(cm):
            print(f"{area_labels[i][:8]:8s} {' '.join([f'{x:10d}' for x in row])}")
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(true_classes, predicted_classes, 
                                      target_names=label_encoder_classes,
                                      output_dict=False)
        print(report)
        
        # Calculate F1 scores
        macro_f1 = f1_score(true_classes, predicted_classes, average='macro')
        weighted_f1 = f1_score(true_classes, predicted_classes, average='weighted')
        
        print(f"\nMacro avg F1: {macro_f1:.3f}")
        print(f"Weighted avg F1: {weighted_f1:.3f}")
    
    # Plot priority distribution
    plt.figure(figsize=(12, 8))
    
    # Calculate average priority score by area
    area_priority_avg = {}
    for area in np.unique(predicted_labels):
        mask = predicted_labels == area
        area_priority_avg[area] = np.mean(priority_scores[mask])
    
    # Sort areas by priority score
    sorted_areas = sorted(area_priority_avg.items(), key=lambda x: x[1], reverse=True)
    areas = [a[0] for a in sorted_areas]
    scores = [a[1] for a in sorted_areas]
    
    # Create bar chart
    bars = plt.bar(areas, scores, color='lightblue')
    
    # Highlight high priority areas
    high_priority_areas = set(predicted_labels[high_priority_indices])
    for i, area in enumerate(areas):
        if area in high_priority_areas:
            bars[i].set_color('darkred')
    
    plt.axhline(high_priority_threshold, color='r', linestyle='--', label='High Priority Threshold')
    plt.title('Average Priority Score by Area')
    plt.xlabel('Area')
    plt.ylabel('Average Priority Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(['High Priority Threshold', 'Regular Areas', 'High Priority Areas'])
    plt.savefig(os.path.join(os.path.dirname(model_dir), 'test_area_priority_scores.png'))
    
    return predicted_labels, priority_scores

if __name__ == "__main__":
    # Fix console encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Define paths
    data_dir = os.environ.get("SELF_HEALING_INTERFERENCE_DATA", os.path.join(os.path.dirname(__file__), "spectrum_management_data", "interference_data"))
    output_dir = os.environ.get("SELF_HEALING_INTERFERENCE_OUTPUT", os.path.join(os.path.dirname(__file__), "outputs", "interference_classifier"))
    
    # Training file
    train_file = os.path.join(data_dir, "spectrum_interference_1.csv")
    
    # Train model
    print("=== Training Model ===")
    model = train_model(train_file, output_dir, epochs=50)
    
    # Test model on different dataset
    print("\n=== Testing Model ===")
    test_file = os.path.join(data_dir, "spectrum_interference_2.csv")
    predicted_labels, priority_scores = test_model(output_dir, test_file)
    
    print("\nModel training and testing complete!")