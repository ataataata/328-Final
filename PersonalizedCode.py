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
    """Load and merge accelerometer and gyroscope data."""
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
    """Calculate magnitude of accelerometer data."""
    data['accel_mag'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
    data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean()  # detrend
    return data

def remove_noise(data, sampling_rate):
    """Apply low-pass filter to remove noise."""
    cutoff = 5  # Hz
    order = 2
    b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
    data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])
    return data

def calc_orientation(data):
    """Calculate pitch and roll angles."""
    data['pitch'] = np.arctan2(data['ay'], np.sqrt(data['ax']**2 + data['az']**2))
    data['roll'] = np.arctan2(-data['ax'], data['az'])    
    data['pitch'] = np.degrees(data['pitch'])
    data['roll'] = np.degrees(data['roll'])
    return data

def calc_tilt_angle(data):
    """Calculate tilt angle."""
    data['tilt_angle'] = np.degrees(np.arccos(data['az'] / (np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2) + 1e-9)))
    return data

def add_features(window):
    """Calculate basic statistical features from a window of data."""
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
    """Extract all features from the sensor data."""
    # Compute accel magnitude, remove noise, and calculate orientation
    data = calc_magnitude(data)
    data = remove_noise(data, sample_rate)
    data = calc_orientation(data)
    data = calc_tilt_angle(data)
    
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
            
            # Tilt angle features
            tilt_feats = {
                'tilt_mean': window['tilt_angle'].mean(),
                'tilt_std': window['tilt_angle'].std()
            }
            
            # Stability and movement intensity
            mean_val = window['filtered_accel_mag'].mean() if window['filtered_accel_mag'].mean() != 0 else 1e-9
            stability = window['filtered_accel_mag'].std() / mean_val
            movement_intensity = np.sum(np.abs(np.diff(window['filtered_accel_mag'])))
            
            extra_feats = {
                'stability': stability,
                'movement_intensity': movement_intensity
            }
            
            combined_features = {**base_features.iloc[0], **orientation_feats, **tilt_feats, **extra_feats}
            combined_features['position'] = position
            
            features_df = pd.concat([features_df, pd.DataFrame([combined_features])], ignore_index=True)
    
    return features_df

def analyze_individual_data(accel_root, gyro_root, user_name, window_sec=5, sample_rate=100):
    """Analyze feature importance for a specific user's data."""
    all_data = pd.DataFrame()
    
    for position in ['back', 'side', 'stomach']:
        # Modified file search pattern to match exact directory structure
        accel_path = os.path.join(accel_root, position, user_name)
        
        # Check if the directory exists
        if os.path.exists(accel_path):
            # Get all CSV files in the directory
            accel_files = glob.glob(os.path.join(accel_path, '*.csv'))
            
            for accel_file in accel_files:
                filename = os.path.basename(accel_file)
                gyro_file = os.path.join(gyro_root, position, user_name, filename)
                
                if not os.path.exists(gyro_file):
                    print(f"No matching gyro file for {accel_file}, skipping.")
                    continue
                    
                print(f"Processing {user_name}'s data for position: {position}")
                merged_data = load_and_merge_data(accel_file, gyro_file, sample_rate)
                features_df = extract_combined_features(merged_data, window_sec, sample_rate, position)
                all_data = pd.concat([all_data, features_df], ignore_index=True)
        else:
            print(f"Directory not found: {accel_path}")
    
    if all_data.empty:
        print(f"No data found for user {user_name}")
        return
    
    # Analyze for each mode
    modes = ["accel", "gyro", "all"]
    results = {}
    
    for mode in modes:
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
                'tilt_mean', 'tilt_std',
                'stability', 'movement_intensity'
            ]
        else:  # accel
            features = [
                'avg', 'max', 'med', 'min', 'q25', 'q75', 'std', 
                'tilt_mean', 'tilt_std'
            ]
        
        # Prepare data for training
        X = all_data[features]
        y = all_data['position']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        dt_model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=6,
            min_samples_leaf=5
        ).fit(X_train, y_train)
        
        # Evaluate
        dt_pred = dt_model.predict(X_test)
        acc = accuracy_score(y_test, dt_pred)
        dt_cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)
        
        # Create feature importance visualization
        importances = dt_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title(f'Feature Importance for {user_name} ({mode.capitalize()} Mode)')
        plt.tight_layout()
        plt.savefig(f"feature_importance_{user_name}_{mode}.png")
        plt.close()
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        position_labels = all_data['position'].unique()
        sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=position_labels, yticklabels=position_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {user_name} ({mode.capitalize()} Mode)')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{user_name}_{mode}.png")
        plt.close()
        
        results[mode] = {
            'accuracy': acc,
            'feature_importance': fi_df.to_dict(),
            'confusion_matrix': dt_cm.tolist()
        }
        
        print(f"\nResults for {user_name} - {mode} mode:")
        print(f"Accuracy: {acc:.3f}")
        print("\nTop 3 most important features:")
        print(fi_df.head(3))
        print("\n" + "="*50)
    
    return results

def main():
    """Main function to run the analysis."""
    # Set paths - modified to match your directory structure
    accel_root = './data/acceloremeter'
    gyro_root = './data/gyroscope'
    
    # List of users to analyze
    users = ['Ata', 'Ceren', 'Emir']
    
    # Process each user's data
    all_results = {}
    for user in users:
        print(f"\nAnalyzing data for {user}")
        results = analyze_individual_data(accel_root, gyro_root, user)
        if results:  # Only store results if data was found
            all_results[user] = results
        
    # Save results to file if we have any
    if all_results:
        with open('sleep_position_analysis_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)
    else:
        print("\nNo results to save - no data was processed.")

if __name__ == "__main__":
    main()