from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from . import config

def build_lite_cnn(input_shape=config.INPUT_SHAPE):
    """
    Builds the Lite-CNN model described in the research.
    Input Shape: (260, 1)
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Layer 1
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Layer 2
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten
        Flatten(),
        
        # Dense
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        # Output
        Dense(2, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = build_lite_cnn()
    model.summary()
