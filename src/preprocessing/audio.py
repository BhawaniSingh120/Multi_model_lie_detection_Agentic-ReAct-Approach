import moviepy as mp
from moviepy import VideoFileClip
import whisper
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from .. import config

# Ensure nltk data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AudioPreprocessor:
    def __init__(self, max_features=130):
        # Load whisper model (base)
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Using TfidfVectorizer with max_features to map to 130 dimensions directly
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.is_fitted = False

    def extract_audio(self, video_path, output_audio_path):
        """Extract audio from video."""
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_audio_path, logger=None)
            return True
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return False

    def transcribe(self, audio_path):
        """Transcribe audio to text."""
        result = self.model.transcribe(audio_path)
        return result['text']

    def clean_text(self, text):
        """Lowercase, remove special chars, remove stopwords, stem."""
        text = text.lower()
        # Remove special chars
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        # Remove stopwords and stem
        cleaned = [self.stemmer.stem(w) for w in words if w not in self.stop_words]
        return " ".join(cleaned)

    def fit_vectorizer(self, corpus):
        """Fit TF-IDF on a list of texts."""
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def get_features(self, text):
        """Transform text to TF-IDF vector."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet.")
        # Returns sparse matrix, convert to array
        vec = self.vectorizer.transform([text]).toarray()
        return vec.flatten() # 130 dimensions

if __name__ == "__main__":
    # Test
    # ap = AudioPreprocessor()
    # ap.fit_vectorizer(["hello world", "this is a test"])
    # v = ap.get_features("hello world")
    # print(v.shape)
    pass
