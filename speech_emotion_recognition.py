import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def mfcc_values(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def delta_values(file_name):
    mfcc = mfcc_values(file_name)
    delta_mfcc = librosa.feature.delta(mfcc)
    return delta_mfcc

def log_mel_values(file_path, duration=3, offset=0.5, n_mels=128):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_mean = np.mean(log_mel_spec.T, axis=0)
    return log_mel_mean

def zcr_values(file_name):
    sig, sr = librosa.load(file_name, duration=3, offset=0.5)
    zcr = np.mean(librosa.feature.zero_crossing_rate(sig).T, axis=0)
    return zcr

def spectral_features(file_path, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_features = np.hstack([centroid, bandwidth, rolloff, flatness, contrast])
    return spectral_features

def extract_features(file_name):
    mfcc = mfcc_values(file_name)
    delta = delta_values(file_name)
    log_mel = log_mel_values(file_name)
    zcr = zcr_values(file_name)
    spectral = spectral_features(file_name)
    all_features = np.hstack([mfcc, delta, log_mel, zcr, spectral])
    return all_features

def main():
    print("🎙️ Starting Speech Emotion Recognition Pipeline...")
    data_dir = "data/"
    audio_files = []
    
    # Traverse directory to find all wav files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
                
    print(f"Found {len(audio_files)} audio files.")
    
    if len(audio_files) == 0:
        print("Error: No audio files found in 'data/' directory.")
        return

    features = []
    labels = []
    
    print("Extracting features... This may take a few minutes.")
    for idx, path in enumerate(audio_files):
        filename = os.path.basename(path)
        # RAVDESS format: 03-01-06-01-02-01-12.wav
        parts = filename.split('-')
        if len(parts) == 7:
            emotion_code = parts[2]
            label = EMOTION_LABELS.get(emotion_code, 'unknown')
            if label != 'unknown':
                try:
                    feat = extract_features(path)
                    features.append(feat)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(audio_files)} files...")

    X = np.array(features)
    y = np.array(labels)
    
    print(f"Feature extraction complete! Extracted {X.shape[1]} features for {X.shape[0]} samples.")
    
    # Save features
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv("features_dataset.csv", index=False)
    print("Saved features to 'features_dataset.csv'.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")
    
    # Handle Imbalance
    print("Handling class imbalance with OverSampling...")
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y_encoded)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    # Train model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('Confusion Matrix - Emotion Recognition')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix plot to 'confusion_matrix.png'.")
    
    # Save model
    joblib.dump(clf, "RandomForest_emotion_model.pkl")
    print("Pipeline complete! Model saved as 'RandomForest_emotion_model.pkl'.")

if __name__ == "__main__":
    main()
