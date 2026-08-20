import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from interference_classifier import load_and_prepare_data, create_sequences, calculate_priority_score

def test_interference_classifier(model_dir, test_file):
    """Test the trained CNN model for area-based interference classification"""
    # Load model and label encoder classes
    model_path = os.path.join(model_dir, 'interference_classifier_cnn.keras')
    label_encoder_path = os.path.join(model_dir, 'label_encoder_classes.npy')
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    label_encoder_classes = np.load(label_encoder_path, allow_pickle=True)
    
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
        
        # Handle potential unknown labels
        valid_mask = np.isin(true_labels, label_encoder_classes)
        if not np.all(valid_mask):
            print(f"\nWarning: {np.sum(~valid_mask)} unknown labels found in test data")
            true_labels = true_labels[valid_mask]
            predicted_labels = predicted_labels[valid_mask]
            
            # Recalculate high priority areas after filtering
            high_priority_areas = predicted_labels[high_priority_indices]
            high_priority_indices = np.where(np.isin(predicted_labels, high_priority_areas))[0]
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(true_labels, predicted_labels, 
                                      labels=label_encoder_classes,
                                      zero_division=0)
        print(report)
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, predicted_labels, 
                             labels=label_encoder_classes)
        
        # Print confusion matrix with area labels
        print("Rows: True Areas, Columns: Predicted Areas")
        area_labels = label_encoder_classes
        # Print column headers
        print("      " + " ".join([f"{area[:8]:10s}" for area in area_labels]))
        for i, row in enumerate(cm):
            print(f"{area_labels[i][:8]:8s} {' '.join([f'{x:10d}' for x in row])}")
    
    # Plot priority distribution by area
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
    high_priority_areas_set = set(predicted_labels[high_priority_indices])
    for i, area in enumerate(areas):
        if area in high_priority_areas_set:
            bars[i].set_color('darkred')
    
    plt.axhline(high_priority_threshold, color='r', linestyle='--', label='High Priority Threshold')
    plt.title('Average Priority Score by Area')
    plt.xlabel('Area')
    plt.ylabel('Average Priority Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(['High Priority Threshold', 'Regular Areas', 'High Priority Areas'])
    
    # Save the plot
    output_dir = os.path.dirname(test_file)
    plt.savefig(os.path.join(output_dir, 'test_area_priority_scores.png'))
    
    return predicted_labels, priority_scores, high_priority_areas_set

if __name__ == "__main__":
    # Fix console encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Define paths
    model_dir = os.environ.get("SELF_HEALING_INTERFERENCE_OUTPUT", os.path.join(os.path.dirname(__file__), "outputs", "interference_classifier"))
    test_file = os.environ.get("SELF_HEALING_INTERFERENCE_TEST_FILE", os.path.join(os.path.dirname(__file__), "spectrum_management_data", "interference_data", "spectrum_interference_3.csv"))
    
    # Test the model
    print("=== Testing Interference Classifier ===")
    predicted_labels, priority_scores, high_priority_areas = test_interference_classifier(model_dir, test_file)
    
    print("\nHigh priority areas that need spectrum allocation:")
    for area in sorted(high_priority_areas):
        print(f"- {area}")
    
    print("\nTesting complete!")