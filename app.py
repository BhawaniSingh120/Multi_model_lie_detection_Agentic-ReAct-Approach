import streamlit as st
import os
import numpy as np
import tensorflow as tf
import tempfile
import cv2
import pickle
from scipy.signal import resample
from src.preprocessing.visual import VisualPreprocessor
from src.preprocessing.audio import AudioPreprocessor
from src import config

# Page Config
st.set_page_config(page_title="Lie Detection System", page_icon="🕵️", layout="wide")

st.title("🕵️ Multimodal Lie Detection System")
st.markdown("Analyze video files to detect Truthfulness vs. Deception using hybrid Visual and Linguistic features.")

# Sidebar
st.sidebar.header("System Status")
model_path = os.path.join(config.MODELS_DIR, "final_model.h5")
vec_path = os.path.join(config.DATA_DIR, "tfidf_vectorizer.pkl")

if os.path.exists(model_path):
    st.sidebar.success(f"Model loaded: {os.path.basename(model_path)}")
else:
    st.sidebar.error("Model not found! Run training.")
    
if os.path.exists(vec_path):
    st.sidebar.success("Vectorizer loaded.")
else:
    st.sidebar.error("Vectorizer not found! Run preprocessing.")

# Main Interface
uploaded_file = st.file_uploader("Upload a Video (MP4/MOV)", type=["mp4", "mov", "avi"])

@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model(model_path)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_resources()

if uploaded_file is not None and model is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    st.video(video_path)
    
    if st.button("Analyze Video"):
        with st.spinner("Processing Video... This feature extraction mimics the research paper pipeline."):
            progress_bar = st.progress(0)
            
            # 1. Init Preprocessors
            vp = VisualPreprocessor()
            ap = AudioPreprocessor(max_features=130)
            # Inject loaded vectorizer
            ap.vectorizer = vectorizer
            ap.is_fitted = True
            
            # 2. Audio Processing (Text)
            st.text("Step 1/3: Audio Extraction & Transcription...")
            temp_audio = "temp_app_audio.wav"
            success = ap.extract_audio(video_path, temp_audio)
            
            text_vec = np.zeros(130)
            transcript_text = "N/A"
            
            if success:
                try:
                    transcript_text = ap.transcribe(temp_audio)
                    clean_text = ap.clean_text(transcript_text)
                    text_vec = ap.get_features(clean_text)
                except Exception as e:
                    st.warning(f"Audio processing failed: {e}")
                finally:
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
            progress_bar.progress(33)
            
            # 3. Visual Processing
            st.text("Step 2/3: Visual Feature Extraction (MobileNetV2)...")
            # Extract frames
            frames = vp.extract_frames(video_path, interval=10) # Faster
            
            # For demo, take mean of frames to get single instance or process as sequence
            # The model expects (260, 1) which is ONE instance.
            # But earlier we concatenated visuals + text. 
            # In `preprocessor.py`, we did it PER FRAME.
            # Here we must aggregate or classify frames.
            # Let's classify each frame and vote?
            # Or replicate the preprocessor logic: Windowing?
            # The preprocessor generated (N, 260) samples. One sample = 1 frame + aligned text.
            
            visual_vectors = []
            for f in frames:
                f_proc = vp.preprocess_frame(f)
                lm = vp.get_landmarks(f_proc) # 130D
                if lm is not None:
                    visual_vectors.append(lm)
            
            progress_bar.progress(66)
            
            if len(visual_vectors) == 0:
                st.error("No face detected in video.")
            else:
                st.text("Step 3/3: Prediction...")
                visual_vectors = np.array(visual_vectors)
                
                # Combine each visual frame with the SAME text vector (global context)
                # (Or temporally aligned if we had timestamps, global is safer for demo)
                
                X_batch = []
                for vis_vec in visual_vectors:
                    norm_vis = vis_vec
                    # Fuse
                    fused = np.concatenate([norm_vis, text_vec])
                    # Normalize
                    f_norm = np.linalg.norm(fused)
                    if f_norm > 0: fused /= f_norm
                    X_batch.append(fused)
                
                X_batch = np.array(X_batch)
                X_batch = np.expand_dims(X_batch, axis=-1) # (N, 260, 1)
                
                # Predict
                preds = model.predict(X_batch)
                # preds shape (N, 2)
                
                # Average probabilities
                avg_pred = np.mean(preds, axis=0) # [prob_truth, prob_lie]
                lie_prob = avg_pred[1]
                
                # Result
                progress_bar.progress(100)
                
                st.write("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction")
                    if lie_prob > 0.5:
                        st.error(f"⚠️ DECEPTION DETECTED ({lie_prob*100:.1f}%)")
                    else:
                        st.success(f"✅ TRUTHFUL ({ (1-lie_prob)*100:.1f}%)")
                        
                with col2:
                    st.subheader("Metadata")
                    st.caption(f"Frames Analyzed: {len(X_batch)}")
                    st.caption("Transcript:")
                    st.write(f"*{transcript_text}*")
                
                # Feature Visualization (Demo)
                st.write("---")
                st.subheader("Feature Analysis")
                st.caption("Average input signature (Visual 0-129 | Text 130-259)")
                avg_feat = np.mean(X_batch, axis=0).flatten()
                st.bar_chart(avg_feat)

    # Cleanup video
    if os.path.exists(video_path):
        os.remove(video_path)
