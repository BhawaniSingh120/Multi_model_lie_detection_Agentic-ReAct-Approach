# Multi-Modal Lie Detection System 🕵️‍♂️

A machine learning system designed to detect truthfulness versus deception in video content by analyzing both visual (facial expressions) and linguistic (audio transcription) features. This project implements a **Lite-CNN** architecture and provides a user-friendly web interface via **Streamlit**.

## 🚀 Features

- **Multi-Modal Analysis**: Combines visual facial features and linguistic text features for robust prediction.
- **Visual Feature Extraction**: Uses facial landmark detection (MobileNetV2/Dlib) to analyze facial expressions.
- **Audio Transcription**: Utilizes **OpenAI Whisper** for accurate speech-to-text transcription.
- **Deep Learning Model**: Custom **Lite-CNN** (Convolutional Neural Network) trained on multimodal data.
- **Interactive Web UI**: Easy-to-use interface for uploading videos and viewing real-time analysis results.
- **Interpretability**: Includes mechanisms to analyze feature importance.

## 📂 Datasets

The system relies on the following datasets, which should be placed in the `raw_data/` directory:

1. **Real-life Deception Detection 2016**
   - **Path**: `raw_data/RealLifeDeceptionDetection.2016.zip`
   - **Description**: A standard benchmark dataset consisting of video clips from real courtroom trials, labeled for truthfulness and deception.

2. **Custom Dataset (dataset2)**
   - **Path**: `raw_data/dataset2.zip`
   - **Description**: An additional proprietary or supplementary dataset used to expand the training distribution and improve model robustness.

## �🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd Multi_Modal_Lie_Detection_System
    ```

2.  **Set Up a Virtual Environment** (Recommended)
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have `ffmpeg` installed on your system if you encounter issues with audio extraction.*

## 🖥️ Usage

### Running the Web Application
To start the lie detection interface:

```bash
streamlit run app.py
```

-   Upload a video file (`.mp4`, `.mov`, `.avi`).
-   The system will process the video, extract features, and display the probability of Truth vs. Deception.

### Running the Full Pipeline
The project includes a batch script (`run_all.bat`) to execute the entire pipeline (Preprocessing -> Training -> Interpretation -> App).

To run the pipeline manually or retrain the model:

1.  **Preprocessing**: Extracts features from the raw dataset.
    ```bash
    python -m src.preprocessor
    ```
2.  **Training**: Runs 5-Fold Cross-Validation.
    ```bash
    python -m src.train
    ```
3.  **Final Training**: Trains the final model on all available data.
    ```bash
    python -m src.train_final
    ```
4.  **Interpretation**: Generates feature importance analysis.
    ```bash
    python -m src.interpret
    ```

## 📂 Project Structure

```
Multi_Modal_Lie_Detection_System/
├── app.py                # Main Streamlit application entry point
├── run_all.bat           # Script to run the full training/inference pipeline
├── requirements.txt      # Python dependencies
├── src/                  # Source code
│   ├── config.py         # Configuration settings (paths, hyperparameters)
│   ├── model.py          # Lite-CNN model architecture definition
│   ├── preprocessor.py   # Data preprocessing logic
│   ├── train.py          # Training script (Cross-Validation)
│   ├── train_final.py    # Training script (Final Model)
│   ├── interpret.py      # Feature interpretation script
│   └── preprocessing/    # Sub-modules for Visual/Audio processing
├── models/               # Directory for saving trained models
├── data/                 # Directory for processed data and artifacts
└── raw_data/             # Directory for original datasets
```

## 🏗️ Technology Stack

-   **Frontend**: Streamlit
-   **Machine Learning**: TensorFlow / Keras
-   **Computer Vision**: OpenCV, Dlib provided logic (via MobileNetV2 in code context)
-   **Audio Processing**: OpenAI Whisper, MoviePy
-   **Data Processing**: NumPy, Pandas, Scikit-learn



