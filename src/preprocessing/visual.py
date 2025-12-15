import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
from .. import config

class VisualPreprocessor:
    def __init__(self):
        # Face Detector (Haar Cascade is fastest/built-in)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        # Feature Extractor: MobileNetV2
        self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        
        # Random Projection Matrix (1280 -> 130)
        # Fix seed for consistency
        np.random.seed(42)
        self.projection_matrix = np.random.randn(1280, 130).astype(np.float32)
        # Normalize matrix?
        self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0)

    def extract_frames(self, video_path, interval=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frames.append(frame)
            count += 1
        cap.release()
        return frames

    def preprocess_frame(self, frame):
        """Histogram Equalization (Y channel of YUV)."""
        if frame is None: return None
        try:
            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img_output
        except:
            return frame

    def get_landmarks(self, frame):
        """
        Detect face, crop, get CNN embedding (1280D), project to 130D.
        Returns flattened 130D vector.
        """
        if frame is None: return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
             return None
        
        # Take largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Crop
        face_img = frame[y:y+h, x:x+w]
        
        # Resize to 224x224 for MobileNet
        try:
            face_img = cv2.resize(face_img, (224, 224))
        except:
            return None
            
        # Preprocess for MobileNet
        face_arr = img_to_array(face_img)
        face_arr = np.expand_dims(face_arr, axis=0)
        face_arr = preprocess_input(face_arr)
        
        # Inference
        features = self.model.predict(face_arr, verbose=0) # Shape (1, 1280)
        features = features.flatten()
        
        # Project
        projected = np.dot(features, self.projection_matrix) # (130,)
        
        return projected

    def process_video(self, video_path):
        frames = self.extract_frames(video_path, interval=10)
        vectors = []
        for f in frames:
            f_proc = self.preprocess_frame(f)
            lm = self.get_landmarks(f_proc)
            if lm is not None:
                vectors.append(lm)
        
        if len(vectors) == 0:
            return np.zeros((1, 130))
            
        return np.array(vectors)

if __name__ == "__main__":
    pass
