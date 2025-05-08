import os
import sys
import librosa
import numpy as np
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)

def extract_features(audio_path):
    """Extract audio features using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)  # Resample to 22050 Hz
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Calculate statistics over features
        features = []
        for feature in [mfcc, chroma, mel, contrast, tonnetz]:
            features.extend([
                np.mean(feature, axis=1),
                np.std(feature, axis=1),
                np.max(feature, axis=1),
                np.min(feature, axis=1)
            ])
        
        # Flatten and normalize
        features = np.concatenate(features)
        features = (features - np.mean(features)) / np.std(features)
        
        return features.tolist()
    
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        return None

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """API endpoint to process uploaded audio"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return jsonify({"error": "Invalid file format"}), 400
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio_file.save(tmp.name)
            features = extract_features(tmp.name)
        
        os.unlink(tmp.name)
        
        if features is None:
            return jsonify({"error": "Failed to extract features"}), 500
        
        return jsonify({"features": features})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
