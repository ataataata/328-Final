import pandas as pd
import glob
import re
import os
import sys
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns

def load_and_merge_data(accel_file, gyro_file, sample_rate):
    # Load accelerometer data
    accel_data = pd.read_csv(accel_file)
    gyro_data = pd.read_csv(gyro_file)
    
    # Convert time to datetime
    accel_data['timestamp'] = pd.to_datetime(accel_data['time'], unit='ns')
    gyro_data['timestamp'] = pd.to_datetime(gyro_data['time'], unit='ns')
    
    accel_data.set_index('timestamp', inplace=True)
    gyro_data.set_index('timestamp', inplace=True)
    
    # Rename columns for consistency
    accel_data.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'}, inplace=True)
    gyro_data.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'}, inplace=True)
    
    # Resample both to ensure same frequency and align times
    accel_data = accel_data.resample(f'{1/sample_rate}s').mean().interpolate()
    gyro_data = gyro_data.resample(f'{1/sample_rate}s').mean().interpolate()
    
    # Merge on nearest timestamps
    merged = pd.merge_asof(accel_data.sort_index(), gyro_data.sort_index(),
                          left_index=True, right_index=True, direction='nearest')
    
    return merged

def calc_magnitude(data):
    # Calculate magnitude using accelerometer data
    data['accel_mag'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
    data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean()  # detrend: "remove gravity"
    return data

def remove_noise(data, sampling_rate):
    # Low pass filter
    cutoff = 5  # Hz
    order = 2
    b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
    data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])
    return data

def calc_orientation(data):
    data['pitch'] = np.arctan2(data['ay'], np.sqrt(data['ax']**2 + data['az']**2))
    data['roll'] = np.arctan2(-data['ax'], data['az'])    
    data['pitch'] = np.degrees(data['pitch'])
    data['roll'] = np.degrees(data['roll'])
    return data

def add_features(window):
    features = {}
    features['avg'] = window['filtered_accel_mag'].mean()
    features['max'] = window['filtered_accel_mag'].quantile(1)
    features['med'] = window['filtered_accel_mag'].quantile(0.5)
    features['min'] = window['filtered_accel_mag'].quantile(0)
    features['q25'] = window['filtered_accel_mag'].quantile(0.25)
    features['q75'] = window['filtered_accel_mag'].quantile(0.75)
    features['std'] = window['filtered_accel_mag'].std()
    return pd.DataFrame([features])

def extract_combined_features(data, window_sec, sample_rate, position):
    # Compute accel magnitude and remove noise
    data = calc_magnitude(data)
    data = remove_noise(data, sample_rate)
    
    # Compute orientation
    data = calc_orientation(data)
    
    # Add gyro magnitude
    data['gyro_mag'] = np.sqrt(data['gx']**2 + data['gy']**2 + data['gz']**2)
    
    # Resample into windows
    window_size = f'{window_sec}s'
    resampled_data = data.resample(window_size)
    
    features_df = pd.DataFrame()
    
    for timestamp, window in resampled_data:
        if not window.empty:
            # Basic accel features
            base_features = add_features(window)
            
            # Orientation features
            orientation_feats = {
                'pitch_mean': window['pitch'].mean(),
                'pitch_std': window['pitch'].std(),
                'roll_mean': window['roll'].mean(),
                'roll_std': window['roll'].std()
            }
            
            # Stability and movement intensity
            mean_val = window['filtered_accel_mag'].mean() if window['filtered_accel_mag'].mean() != 0 else 1e-9
            stability = window['filtered_accel_mag'].std() / mean_val
            movement_intensity = np.sum(np.abs(np.diff(window['filtered_accel_mag'])))
            
            extra_feats = {
                'stability': stability,
                'movement_intensity': movement_intensity
            }
            
            combined_features = {**base_features.iloc[0], **orientation_feats, **extra_feats}
            combined_features['position'] = position
            
            features_df = pd.concat([features_df, pd.DataFrame([combined_features])], ignore_index=True)
    
    return features_df

def process_combined_data(accel_root, gyro_root, output_filename="combined_sleep_data.csv", window_sec=5, sample_rate=100):
    all_data = pd.DataFrame()
    
    for position in ['back', 'side', 'stomach']:
        accel_files = glob.glob(os.path.join(accel_root, position, '*.csv'))
        
        for accel_file in accel_files:
            filename = os.path.basename(accel_file)
            gyro_file = os.path.join(gyro_root, position, filename)
            
            if not os.path.exists(gyro_file):
                print(f"No matching gyro file for {accel_file}, skipping.")
                continue
                
            merged_data = load_and_merge_data(accel_file, gyro_file, sample_rate)
            features_df = extract_combined_features(merged_data, window_sec, sample_rate, position)
            all_data = pd.concat([all_data, features_df], ignore_index=True)
    
    all_data.to_csv(output_filename, index=False)
    return all_data

def train_position_classifier(frames, mode="all"):
    # Select features based on mode
    if mode == "gyro":
        features = [
            'pitch_mean', 'pitch_std', 'roll_mean', 'roll_std',
            'stability', 'movement_intensity'
        ]
    elif mode == "all":
        features = [
            'avg', 'max', 'med', 'min', 'q25', 'q75', 'std',
            'pitch_mean', 'pitch_std', 'roll_mean', 'roll_std',
            'stability', 'movement_intensity'
        ]        
    elif mode == "accel":
        features = [
            'avg', 'max', 'med', 'min', 'q25', 'q75', 'std'
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'gyro', 'accel', or 'all'")
    
    X = frames[features]
    y = frames['position']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    dt_model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_leaf=5
    ).fit(X_train, y_train)
    
    dt_pred = dt_model.predict(X_test)
    
    acc = accuracy_score(y_test, dt_pred)
    print(f"\nResults for {mode} mode:")
    print(classification_report(y_test, dt_pred))
    print("Accuracy on test set:", acc)
    
    dt_cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)
    
    return dt_model, dt_cm, acc
# Example usage
if __name__ == "__main__":
    modes = ["accel", "gyro", "all"]  # Modes to test
    accel_root = './data/acceloremeter'
    gyro_root = './data/gyroscope'
    window_sizes = [1, 5, 15]  # Window sizes to compare
    
    results = []  # To store results for each configuration
    
    for window_sec in window_sizes:
        for mode in modes:
            output_file = f"combined_sleep_data_{mode}_{window_sec}s.csv"
            
            # Process data and extract features
            all_data = process_combined_data(accel_root, gyro_root, output_file, window_sec=window_sec, sample_rate=100)
            
            # Train the classifier using the selected mode
            dt_model, dt_cm, acc = train_position_classifier(all_data, mode=mode)
            
            # Save results
            results.append({
                'window_sec': window_sec,
                'mode': mode,
                'accuracy': acc,
                'confusion_matrix': dt_cm,
                'model': dt_model
            })
            
            # Visualize confusion matrix
            position_labels = all_data['position'].unique()
            plt.figure(figsize=(6, 4))
            sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=position_labels, yticklabels=position_labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for Sleep Position Classification ({mode.capitalize()} Mode, {window_sec}s Window)')
            plt.savefig(f"confusion_matrix_{mode}_{window_sec}s.png")
            
            # Select features based on mode
            if mode == "gyro":
                features = [
                    'pitch_mean', 'pitch_std', 'roll_mean', 'roll_std',
                    'stability', 'movement_intensity'
                ]
            elif mode == "all":
                features = [
                    'avg', 'max', 'med', 'min', 'q25', 'q75', 'std',
                    'pitch_mean', 'pitch_std', 'roll_mean', 'roll_std',
                    'stability', 'movement_intensity'
                ]
            elif mode == "accel":
                features = [
                    'avg', 'max', 'med', 'min', 'q25', 'q75', 'std'
                ]
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'gyro', 'accel', or 'all'")
            
            # Get feature importances
            importances = dt_model.feature_importances_
            
            # Verify feature length
            if len(features) != len(importances):
                print(f"Number of features ({len(features)}) doesn't match importance values ({len(importances)})")
                raise ValueError("Feature length mismatch")
            
            # Create and sort DataFrame
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            fi_df = fi_df.sort_values('Importance', ascending=False)
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df)
            plt.title(f'Feature Importance ({mode.capitalize()} Mode, {window_sec}s Window)')
            plt.savefig(f"feature_importance_{mode}_{window_sec}s.png")
