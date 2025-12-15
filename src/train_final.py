import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from . import config
from .model import build_lite_cnn

def train_final_model():
    print("Loading full dataset...")
    try:
        X = np.load(os.path.join(config.DATA_DIR, "X.npy"))
        y = np.load(os.path.join(config.DATA_DIR, "y.npy"))
    except FileNotFoundError:
        print("Data not found. Run preprocessor first.")
        return

    print(f"Data Shape: X={X.shape}, y={y.shape}")
    
    # Build Model
    model = build_lite_cnn(input_shape=(260, 1))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define save path
    save_path = os.path.join(config.MODELS_DIR, "final_model.h5")
    
    # Callbacks
    # Since we have no validation set (using all data), we save the model at the end
    # Or we can save the best training accuracy, but that risks overfitting.
    # Usually for final production model, we train for fixed epochs or until loss is low.
    # We will save at the end.
    
    print(f"Training on all data for {config.EPOCHS} epochs...")
    
    history = model.fit(
        X, y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    model.save(save_path)
    print(f"Final model saved to {save_path}")

if __name__ == "__main__":
    train_final_model()
