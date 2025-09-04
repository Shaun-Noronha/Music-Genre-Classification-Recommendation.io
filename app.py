import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

# Genres based on your notebook’s mapping
CLASS_NAMES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

MODEL_PATH = Path("model_4_complete.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y)
    target_len = sr * 30
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Recreate the same features as in training CSV
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    harmony = np.mean(np.abs(librosa.effects.harmonic(y)))
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    tempi = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
    tempo = float(np.median(tempi)) if tempi is not None and len(tempi) else 0.0
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)

    feats = np.hstack([
        chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr,
        harmony, flatness, tempo, mfccs
    ]).astype(np.float32)
    return feats

def predict(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    feats = extract_features(y, sr)[None, :]
    probs = model.predict(feats, verbose=0)[0]
    idxs = probs.argsort()[-3:][::-1]
    return {CLASS_NAMES[i]: float(probs[i]) for i in idxs}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload a 5–30s clip (wav/mp3)"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 Genres"),
    title="Music Genre Classification",
    description="Upload an audio clip and get predicted genres."
)

if __name__ == "__main__":
    demo.launch()
