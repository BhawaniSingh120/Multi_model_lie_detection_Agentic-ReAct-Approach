import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from scipy.signal import resample
from .preprocessing.visual import VisualPreprocessor
from .preprocessing.audio import AudioPreprocessor
from . import config

def process_dataset():
    # 1. Load Metadata
    meta_path = os.path.join(config.DATA_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        print("Metadata not found. Run data_loader properly.")
        return
    
    df = pd.read_csv(meta_path)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Initialize Preprocessors
    vp = VisualPreprocessor()
    ap = AudioPreprocessor(max_features=130)
    
    # Storage
    # We need to first collect all text to fit TF-IDF
    print("Step 1: collecting text for TF-IDF...")
    all_texts = []
    video_transcripts = {} # video_path -> segments [{'start':, 'end':, 'text':...}]
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        video_path = row['video_path']
        if not os.path.exists(video_path):
            continue
            
        # Audio extraction
        audio_path = os.path.join(config.DATA_DIR, "temp_audio.wav")
        success = ap.extract_audio(video_path, audio_path)
        if not success:
            continue
            
        try:
            # Transcribe (Whisper returns segments)
            result = ap.model.transcribe(audio_path)
            segments = result['segments']
            
            cleaned_segments = []
            for s in segments:
                clean_t = ap.clean_text(s['text'])
                if clean_t:
                    all_texts.append(clean_t)
                    cleaned_segments.append({
                        'start': s['start'],
                        'end': s['end'],
                        'text': clean_t
                    })
            video_transcripts[video_path] = cleaned_segments
        except Exception as e:
            print(f"Transcription error {video_path}: {e}")
        
        # Cleanup temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    # Fit TF-IDF
    if not all_texts:
        print("No text found!")
        return
        
    print(f"Fitting TF-IDF on {len(all_texts)} segments...")
    ap.fit_vectorizer(all_texts)
    
    # Step 2: Extract Frames and Fuse
    print("Step 2: Extracting visual features and fusing...")
    
    X = []
    y = []
    groups = [] # To store video index for splitting
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        video_path = row['video_path']
        label = row['label']
        if video_path not in video_transcripts:
            continue
            
        transcripts = video_transcripts[video_path]
        
        # Extract Frames (interval=5 approx 5-6 fps)
        # Using a slightly higher interval to reduce data size if needed, but 5 is fine.
        frames = vp.extract_frames(video_path, interval=5)
        
        # FPS estimation (needed for timestamp alignment)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0: fps = 25.0
        
        for i, frame in enumerate(frames):
            # Timestamp
            frame_n = i * 5
            timestamp = frame_n / fps
            
            # Find corresponding text segment
            # Naive: find segment where timestamp roughly falls
            text_vec = np.zeros(130)
            for seg in transcripts:
                if seg['start'] <= timestamp <= seg['end']:
                    text_vec = ap.get_features(seg['text'])
                    break
            
            # Visual Feature
            f_proc = vp.preprocess_frame(frame)
            vis_vec = vp.get_landmarks(f_proc) # 136D
            
            if vis_vec is None:
                continue # Skip frame if no face
            
            # Resize Visual 136 -> 130
            vis_vec_130 = resample(vis_vec, 130)
            
            # Concatenate
            fused = np.concatenate([vis_vec_130, text_vec]) # 260
            
            # Normalize fused vector?
            # Step 4 says "Normalize the combined vector".
            # Simple L2 norm
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
                
            X.append(fused)
            y.append(label)
            groups.append(idx)
            
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"Processed Data Shape: X={X.shape}, y={y.shape}")
    
    # Expand dims for Conv1D: (N, 260, 1)
    X = np.expand_dims(X, axis=-1)
    
    # One-hot encode y? No, SparseCatCrossEntropy or Dense(2) with standard CatCrossEntropy requires one-hot.
    # Step 36 says "Categorical Cross-Entropy". So we need one-hot.
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y, num_classes=2)
    
    # Save
    np.save(os.path.join(config.DATA_DIR, "X.npy"), X)
    np.save(os.path.join(config.DATA_DIR, "y.npy"), y_cat)
    np.save(os.path.join(config.DATA_DIR, "groups.npy"), groups)
    
    # Save Vectorizer
    import pickle
    with open(os.path.join(config.DATA_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(ap.vectorizer, f)
        
    print("Saved processed arrays and vectorizer.")

if __name__ == "__main__":
    process_dataset()
