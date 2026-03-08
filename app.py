import streamlit as st
import librosa
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import librosa.display
from speech_emotion_recognition import extract_features

st.set_page_config(page_title="VaaniAI: Emotion-Aware Speech", page_icon="🎙️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #ff4b4b;
        color: white;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #262730;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_rf_model():
    try:
        model = joblib.load("RandomForest_emotion_model.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except FileNotFoundError:
        return None, None

model, le = load_rf_model()

emotion_emojis = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲',
    'unknown': '❓'
}

st.title("🎙️ VaaniAI: Emotion-Aware Study Buddy")
st.markdown("Upload a **.wav** audio file to detect the student's emotional state. The prototype uses a Random Forest machine learning engine mathematically analyzing MFCCs and Mel-Spectrogram features from spoken audio to detect stress, confusion, or confidence in real-time.")

if model is None or le is None:
    st.error("⚠️ Random Forest model files not found! Please run `speech_emotion_recognition.py` first.")
else:
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        # Save temp file
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.markdown("### Audio Analysis")
        col1, col2 = st.columns([1, 1])
        
        with st.spinner("Analyzing audio and extracting features..."):
            try:
                # Load audio for visualization
                y, sr = librosa.load(temp_path, duration=3, offset=0.5)
                
                with col1:
                    st.markdown("**Waveform**")
                    fig_wave, ax_wave = plt.subplots(figsize=(5, 3))
                    fig_wave.patch.set_facecolor('#0e1117')
                    ax_wave.set_facecolor('#0e1117')
                    ax_wave.tick_params(colors='white')
                    ax_wave.xaxis.label.set_color('white')
                    ax_wave.yaxis.label.set_color('white')
                    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#1f77b4')
                    st.pyplot(fig_wave)
                    
                    st.markdown("**Mel Spectrogram**")
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    log_mel_spec = librosa.power_to_db(mel_spec)
                    fig_mel, ax_mel = plt.subplots(figsize=(5, 3))
                    fig_mel.patch.set_facecolor('#0e1117')
                    img = librosa.display.specshow(log_mel_spec, sr=sr, ax=ax_mel, cmap='magma')
                    st.pyplot(fig_mel)

                with col2:
                    st.markdown("### Classification Results")
                    features = extract_features(temp_path).reshape(1, -1)
                    prediction_encoded = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    predicted_emotion = le.inverse_transform([prediction_encoded])[0]
                    confidence = np.max(probabilities) * 100
                    prob_dict = {le.inverse_transform([i])[0]: prob * 100 for i, prob in enumerate(probabilities)}
                    
                    emoji = emotion_emojis.get(predicted_emotion, '❓')
                    
                    st.markdown(f"""
                    <div class="emotion-box">
                        <h2 style="color:white;">Student Emotion</h2>
                        <h1 style="font-size: 3rem;">{predicted_emotion.capitalize()} {emoji}</h1>
                        <p style="color: #a3a8b8;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>**Emotion Confidence Distribution (%)**", unsafe_allow_html=True)
                    st.bar_chart(prob_dict)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
