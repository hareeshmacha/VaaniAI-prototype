# 🎙️ VaaniAI: Emotion-Aware AI Study Buddy (Prototype)

**VaaniAI** is an intelligent educational technology prototype designed to detect student emotions from speech in real time. 

Current AI tutors lack emotional awareness, often leaving students feeling stressed or confused. We are building a system that understands auditory emotional intelligence to adapt explanations, difficulty, and motivation accordingly. 

This repository contains the **Phase 3 Prototype** for the AI Unlocked challenge.

---

## 🎯 The Concept & Problem
Students often feel stress, confusion, and low motivation while studying. 
VaaniAI detects emotions from student speech and is designed to adapt its tutoring accordingly:
- **Stressed (Fearful/Angry)** → Provide motivation + simpler help
- **Confused (Sad)** → Provide step-by-step guidance
- **Confident (Happy/Calm)** → Offer higher difficulty challenges

*Note: This prototype iteration successfully implements the core **Speech Emotion Recognition (SER)** engine, proving the technical feasibility of extracting and classifying these emotions from raw audio.*

---

## 🚀 The Prototype Pipeline
This prototype runs entirely locally and uses a **Random Forest Ensemble** architecture to ensure fast, lightweight inference suitable for web deployment.

1. **Dataset:** We use the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). We process a balanced subset of **2,452 audio files** containing 8 core emotional expressions.
2. **Feature Engineering:** We use `librosa` to extract high-dimensional acoustic features mathematically proven to correlate with emotion:
   - **MFCCs & Delta MFCCs** ( Mel-Frequency Cepstral Coefficients)
   - **Log-Mel Spectrograms** (Power distribution across frequencies)
   - **Zero Crossing Rate (ZCR)**
   - **Spectral Metadata** (Centroid, Bandwidth, Contrast, Rolloff, Flatness)
3. **Machine Learning Model:** A robust `RandomForestClassifier` (300 estimators) trained on the extracted features, achieving **~85% accuracy** on the hold-out test set.
4. **Web UI:** A real-time, interactive **Streamlit** dashboard where users can upload audio, visualize the spectrogram/waveform, and view the model's emotional prediction and confidence scores.

---

## 🌐 Running it Locally

You can test the prototype on any `.wav` file by starting the Streamlit interface! 

1. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. *(Optional)* If you wish to retrain the model on the RAVDESS dataset yourself, place the dataset into a `data/` folder and run the training pipeline:
   ```bash
   python speech_emotion_recognition.py
   ```
3. Run the interactive Streamlit web application:
   ```bash
   streamlit run app.py
   ```
4. A web browser will automatically open (`http://localhost:8501`) where you can upload audio files, view their waveform visualizations, and get real-time emotion predictions!

---

## � Future Roadmap (The Complete Product)
To evolve this prototype into the scalable, complete "VaaniAI" product, our architectural roadmap includes:
- **Deep Learning Upgrade:** Migrating the Emotion Engine to a PyTorch CNN-LSTM designed for complex time-sequence pattern recognition.
- **Generative AI Integration:** Connecting detected emotions to **Azure OpenAI** to generate dynamic, adaptive text responses for the student.
- **Backend Architecture:** Wrapping the engine in a **FastAPI** backend and tracking student emotional analytics in an **Azure Cosmos DB** for long-term learning insights. 

---

## 📝 Dataset Citation
*Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391.* https://doi.org/10.1371/journal.pone.0196391.
