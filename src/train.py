import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from . import config
from .model import build_lite_cnn

def train_model():
    # Load Data
    print("Loading data...")
    try:
        X = np.load(os.path.join(config.DATA_DIR, "X.npy"))
        y = np.load(os.path.join(config.DATA_DIR, "y.npy"))
        groups = np.load(os.path.join(config.DATA_DIR, "groups.npy"))
    except FileNotFoundError:
        print("Data not found. Run preprocessor first.")
        return

    print(f"Data Shape: X={X.shape}, y={y.shape}")
    
    # 5-Fold Group CV
    gkf = GroupKFold(n_splits=5)
    
    fold = 1
    accuracies = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        print(f"\n--- Fold {fold}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Build Model
        model = build_lite_cnn(input_shape=(260, 1))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(config.MODELS_DIR, f"model_fold_{fold}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        accuracies.append(acc)
        
        fold += 1
        
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    train_model()
