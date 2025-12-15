import numpy as np
import tensorflow as tf
import os
import pickle
import pandas as pd
from . import config

def interpret_model(model_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load Vectorizer
    vec_path = os.path.join(config.DATA_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(vec_path):
        print("Vectorizer not found.")
        return
        
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    feature_names = vectorizer.get_feature_names_out()
    
    # Load Data (Test set ideally, but just load all X)
    X = np.load(os.path.join(config.DATA_DIR, "X.npy"))
    y = np.load(os.path.join(config.DATA_DIR, "y.npy"))
    
    # Take a sample (e.g. first 100) or average over Lie class
    # Find Lie samples (y[:,1] == 1)
    lie_indices = np.where(y[:,1] == 1)[0]
    if len(lie_indices) > 100:
        lie_indices = lie_indices[:100]
        
    X_sample = X[lie_indices]
    
    # Saliency Map: Gradient of Class 1 score w.r.t Input
    # Input X shape (N, 260, 1)
    images = tf.Variable(X_sample, dtype=float)
    
    with tf.GradientTape() as tape:
        preds = model(images)
        loss = preds[:, 1] # Score for 'Lie'
        
    grads = tape.gradient(loss, images) # (N, 260, 1)
    grads = tf.abs(grads)
    
    # Average importance per feature
    mean_grads = np.mean(grads.numpy(), axis=0).flatten() # (260,)
    
    # Visual Importance (0-129)
    vis_imp = np.mean(mean_grads[:130])
    print(f"Average Visual Feature Importance: {vis_imp:.6f}")
    
    # Text Importance (130-259)
    text_grads = mean_grads[130:]
    text_imp = np.mean(text_grads)
    print(f"Average Text Feature Importance: {text_imp:.6f}")
    
    # Top words
    top_indices = np.argsort(text_grads)[::-1][:10] # Top 10 words
    print("\nTop influential words for 'Lie':")
    for idx in top_indices:
        if idx < len(feature_names):
            print(f"{feature_names[idx]}: {text_grads[idx]:.6f}")
        else:
            print(f"Index {idx} out of bounds")

if __name__ == "__main__":
    # Find list of models
    models = [f for f in os.listdir(config.MODELS_DIR) if f.endswith(".h5")]
    if models:
        # Use first one
        interpret_model(os.path.join(config.MODELS_DIR, models[0]))
    else:
        print("No models found.")
