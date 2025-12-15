import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Dataset paths
REAL_LIFE_ZIP = os.path.join(BASE_DIR, "raw_data", "RealLifeDeceptionDetection.2016.zip")
OWN_DATASET_ZIP = os.path.join(BASE_DIR, "raw_data", "dataset2.zip") 

SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "raw_data", "shape_predictor_68_face_landmarks.dat.bz2")
PREDICTOR_MODEL = os.path.join(BASE_DIR, "raw_data", "shape_predictor_68_face_landmarks.dat")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_SHAPE = (260, 1)

# Feature Config
VISUAL_DIM = 128 # 128D embedding? Or just action units? 
# The paper says: "128-dimensional facial embedding"
# plus 130D for vector? The instructions say "Visual Vector (mapped to 130 dimensions)"
# and "Linguistic Vector (mapped to 130 dimensions)"
# But earlier "128-dimensional facial embedding". 
# Maybe 128 + some other features? Or mapped/projected?
# Step 4 says: "Visual Vector (mapped to 130 dimensions)" and "Linguistic Vector (mapped to 130 dimensions)".
# So we might need a dense layer to project 128 -> 130 if the raw is 128.
# Or maybe the 128 is from OpenFace and they append something?
# Let's assume we get 128 and project to 130 or pad.
# Actually, let's verify later. For now, 260 total.
