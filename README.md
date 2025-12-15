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

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Multi_Modal_Lie_Detection_System

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



