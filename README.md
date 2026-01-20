# BISINDO Sign Language Translator

<<<<<<< HEAD
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A real-time Indonesian Sign Language (BISINDO) recognition system using computer vision and machine learning. This project translates hand gestures into text, supporting both static and dynamic signs for enhanced accessibility.

## ✨ Features

- **Dual Gesture Recognition**: Combines Random Forest for static gestures and LSTM/Transformer models for dynamic sequences
- **Real-time Processing**: Live gesture recognition using webcam or Raspberry Pi camera
- **Cross-platform Deployment**: Works on PC (Windows/Linux) and Raspberry Pi
- **Optimized Models**: TensorFlow Lite conversion for efficient edge deployment
- **Complete Pipeline**: From data collection to model training and inference
- **MediaPipe Integration**: Accurate hand landmark detection with 126 features per frame

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam or Raspberry Pi camera
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Olfatalatas/BISINDO-Sign-Language-Translator.git
   cd BISINDO-Sign-Language-Translator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Raspberry Pi deployment, install TFLite runtime:**
   ```bash
   pip install tflite-runtime
   ```

### Usage

#### Real-time Recognition on PC

Run the combined RF + LSTM inference:
```bash
python 06_inference/Code\ Real\ Time\ PC/test_program_rf_lstm.py
```

Or use Transformer model:
```bash
python 06_inference/Code\ Real\ Time\ PC/test_program_rf_transformer.py
```

#### Real-time Recognition on Raspberry Pi

```bash
python 06_inference/Code\ Real\ Time\ Raspberry\ Pi/realtime_lstm_raspi.py
```

The system will:
- Open camera feed
- Detect hand gestures in real-time
- Display recognized signs with confidence scores
- Output text translations

## 📁 Project Structure

```
BISINDO-Sign-Language-Translator/
├── 01_data_collection/          # Scripts for collecting gesture datasets
│   ├── collect_dynamic_gesture_dataset.py
│   └── collect_static_gesture_dataset.py
├── 02_preprocessing/            # Data preprocessing and augmentation
│   ├── augment_static_dataset.py
│   ├── extract_landmark_static_dataset.py
│   └── merge_labels.py
├── 03_training/                 # Model training scripts
│   ├── train_lstm.py
│   ├── train_RF.py
│   └── train_transformer.py
├── 04_benchmarking_analysis/    # Performance evaluation
│   ├── benchmark_confidence_stress_test.py
│   └── benchmark_speed_train.py
├── 05_model_conversion/         # Convert models to TFLite
│   ├── convert_model_lstm.py
│   └── convert_model_transformer.py
├── 06_inference/                # Real-time inference applications
│   ├── Code Real Time PC/
│   └── Code Real Time Raspberry Pi/
├── models/                      # Trained models and labels
├── sample_data/                 # Sample gesture datasets
└── requirements.txt
```

## 🛠️ Training Your Own Models

1. **Collect Data:**
   ```bash
   python 01_data_collection/collect_static_gesture_dataset.py
   python 01_data_collection/collect_dynamic_gesture_dataset.py
   ```

2. **Preprocess Data:**
   ```bash
   python 02_preprocessing/augment_static_dataset.py
   python 02_preprocessing/extract_landmark_static_dataset.py
   ```

3. **Train Models:**
   ```bash
   python 03_training/train_RF.py
   python 03_training/train_lstm.py
   python 03_training/train_transformer.py
   ```

4. **Convert for Deployment:**
   ```bash
   python 05_model_conversion/convert_model_lstm.py
   python 05_model_conversion/convert_model_transformer.py
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs
- Suggesting features
- Submitting pull requests
- Code style guidelines

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Olfatalatas/BISINDO-Sign-Language-Translator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Olfatalatas/BISINDO-Sign-Language-Translator/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Maintainers

- **Olfat Alatas** - [GitHub](https://github.com/Olfatalatas)

## 🙏 Acknowledgments

- Built with [MediaPipe](https://mediapipe.dev/) for hand tracking
- Powered by [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/)
- Inspired by the BISINDO sign language community

---

*Helping bridge communication gaps through technology* 🤟</content>
<parameter name="filePath">d:\Kuliah\Tugas Akhir\BISINDO-Sign-Language-Translator\README.md
=======
[![License](https://img.shields.io/github/license/Olfatalatas/BISINDO-Sign-Language-Translator)]()
[![Repo Size](https://img.shields.io/github/repo-size/Olfatalatas/BISINDO-Sign-Language-Translator)]()

A research and engineering codebase for collecting, training, converting, benchmarking, and running real-time BISINDO (Indonesian Sign Language) gesture recognition models using MediaPipe, scikit-learn, and TensorFlow (LSTM / Transformer). The project contains data-collection scripts, model training and conversion utilities, benchmarking tools, and inference programs for both PC and Raspberry Pi.

Table of Contents
- [What this project does](#what-this-project-does)
- [Why it's useful](#why-its-useful)
- [Project structure (key files)](#project-structure-key-files)
- [Get started (quickstart)](#get-started-quickstart)
  - [Prerequisites](#prerequisites)
  - [Install dependencies](#install-dependencies)
  - [Prepare dataset (sample & custom)](#prepare-dataset-sample--custom)
  - [Train a model (LSTM / Transformer)](#train-a-model-lstm--transformer)
  - [Convert models to TFLite](#convert-models-to-tflite)
  - [Run real-time inference (PC)](#run-real-time-inference-pc)
  - [Run real-time inference (Raspberry Pi)](#run-real-time-inference-raspberry-pi)
- [Models and data layout](#models-and-data-layout)
- [Where to get help](#where-to-get-help)
- [Who maintains and how to contribute](#who-maintains-and-how-to-contribute)
- [Notes & Tips](#notes--tips)
- [License](#license)

What this project does
----------------------
This repository implements an end-to-end workflow for BISINDO gesture recognition:
- Collect static and dynamic gesture datasets with webcam / Pi Camera (MediaPipe landmarks).
- Train sequence models (LSTM and Transformer) and static classifiers (Random Forest).
- Convert trained models to optimized formats (TensorFlow SavedModel, .h5, and TFLite).
- Benchmark training and inference performance and confidence.
- Provide real-time inference example programs for desktop and Raspberry Pi.

Why it's useful
---------------
- Accelerates prototyping for sign-language recognition systems.
- Provides full pipeline: data collection → training → conversion → real-time inference.
- Includes both static (RF) and temporal (LSTM/Transformer) model examples for hybrid deployment.
- Raspberry Pi-targeted scripts and TFLite conversion for edge deployment.

Project structure (key files)
-----------------------------
- 01_data_collection/
  - collect_static_gesture_dataset.py — capture images for static gestures
  - collect_dynamic_gesture_dataset.py — capture sequences for dynamic gestures
- 02_preprocessing/
  - merge_labels.py — merge label encoders / label arrays into unified labels file
- 03_training/
  - train_lstm.py — training script for LSTM-based dynamic models
  - train_transformer.py — training script for Transformer-based dynamic models
- 04_benchmarking_analysis/
  - benchmark_speed_train.py — benchmark training speed
  - benchmark_confidence_stress_test.py — evaluate confidence / stress tests
- 05_model_conversion/
  - convert_model_lstm.py — convert LSTM .h5 → .tflite (and helpers)
  - (similar helpers for Transformer conversion)
- 06_inference/
  - Code Real Time PC/
    - test_program_rf_lstm.py — realtime hybrid RF + LSTM example (PC)
    - test_program_rf_transformer.py — realtime hybrid RF + Transformer (PC)
  - Code Real Time Raspberry Pi/
    - realtime_lstm_raspi.py — realtime LSTM TFLite example for Raspberry Pi (Picamera2)
    - realtime_transformer_raspi.py — realtime Transformer TFLite example for Raspberry Pi
- models/ — expected location for trained model files and label maps
- sample_data/ — sample dataset directories (dynamic / static)
- LICENSE — repository license
- CONTRIBUTING.md — contribution guidelines (if present)

Get started (quickstart)
------------------------

Prerequisites
- Recommended Python: 3.8 — 3.11
- For training: a machine with sufficient RAM and optional GPU (for TF).
- For Raspberry Pi inference: Raspberry Pi OS, Picamera2 (if using camera), recent TensorFlow Lite build compatible with your Pi.

Install dependencies (example)
- Create and activate virtual environment:
  - python -m venv .venv
  - source .venv/bin/activate  (Linux/macOS) or .venv\Scripts\activate (Windows)
- Install core packages:
  - pip install -U pip
  - pip install numpy opencv-python mediapipe tensorflow scikit-learn joblib matplotlib seaborn

Notes:
- mediapipe installation on Raspberry Pi may require special builds or platform-specific steps.
- For Raspberry Pi camera support, install picamera2:
  - sudo apt install -y python3-picamera2  (or follow Picamera2 docs)

Prepare dataset (sample & custom)
- The repository includes a small sample dataset under sample_data/ for quick tests.
- To collect your own static dataset:
  - Edit parameters (PERSON_NAME, GESTURE_NAME, SAVE_DIR) in:
    - 01_data_collection/collect_static_gesture_dataset.py
  - Run:
    - python 01_data_collection/collect_static_gesture_dataset.py
- To collect dynamic sequences:
  - Edit parameters (PERSON_NAME, SEQUENCE_LENGTH, TOTAL_SEQUENCES) in:
    - 01_data_collection/collect_dynamic_gesture_dataset.py
  - Run:
    - python 01_data_collection/collect_dynamic_gesture_dataset.py

Train a model (LSTM / Transformer)
- Example: LSTM (using sample data)
  - python 03_training/train_lstm.py
  - Outputs are saved to models/ (e.g., best_model_lstm.h5)
- Example: Transformer
  - python 03_training/train_transformer.py
  - Outputs saved to models/ (e.g., transformer_best.keras)
- Training scripts assume dataset located at sample_data/dynamic by default; change DATASET_DIR in script if needed.

Convert models to TFLite
- After training, convert to TFLite for edge inference:
  - python 05_model_conversion/convert_model_lstm.py
  - Conversion scripts write .tflite files to models/ (e.g., model_lstm.tflite, model_transformer.tflite)
- See conversion scripts for options: quantization, float vs int8, representative datasets.

Run real-time inference (PC)
- Ensure models/ contains:
  - random_forest_bisindo_kcross.pkl
  - best_model_lstm.h5 or transformer_best.keras (for PC scripts)
  - or model_lstm.tflite / model_transformer.tflite for TFLite usage
  - all_gestures_labels.npy — merged labels mapping
- Example (PC, LSTM + RF hybrid):
  - python "06_inference/Code Real Time PC/test_program_rf_lstm.py"
- Example (PC, Transformer + RF hybrid):
  - python "06_inference/Code Real Time PC/test_program_rf_transformer.py"
- The PC scripts use a webcam (cv2.VideoCapture(0)) and MediaPipe hands to extract landmarks then predict using RF and sequence model.

Run real-time inference (Raspberry Pi)
- Copy models/ with TFLite artifacts to Raspberry Pi.
- Ensure Picamera2 and dependencies are installed on Pi.
- Example (Pi, LSTM TFLite):
  - python "06_inference/Code Real Time Raspberry Pi/realtime_lstm_raspi.py"
- Example (Pi, Transformer TFLite):
  - python "06_inference/Code Real Time Raspberry Pi/realtime_transformer_raspi.py"

Models and data layout
----------------------
- models/
  - random_forest_bisindo_kcross.pkl — RandomForest classifier for static gestures
  - best_model_lstm.h5 — trained LSTM model (Keras .h5)
  - transformer_best.keras — full Transformer model (SavedModel or Keras)
  - model_lstm.tflite / model_transformer.tflite — TFLite converted models
  - all_gestures_labels.npy — dictionary with keys 'static' and 'dynamic' mapping class indices → gesture names
- sample_data/
  - static/ — example static gesture images (if present)
  - dynamic/ — example dynamic gesture sequences used for training

Where to get help
-----------------
- Open an issue on this repository: Issues → New issue
- Check the code comments in scripts under 01_data_collection, 03_training, 05_model_conversion, 06_inference for usage notes
- If you need platform-specific help (Raspberry Pi / Picamera2 / TFLite), consult:
  - TensorFlow Lite docs: https://www.tensorflow.org/lite
  - MediaPipe docs: https://developers.google.com/mediapipe
  - Picamera2 docs (Raspberry Pi): https://www.raspberrypi.com/documentation/

Who maintains and how to contribute
----------------------------------
- Maintainer: Olfatalatas (GitHub: Olfatalatas)
- To contribute:
  - Fork the repository and open a pull request with a clear description of your changes.
  - For larger changes, open an issue first to discuss the approach.
  - See CONTRIBUTING.md (if present) for detailed contribution guidelines.
- Please keep PRs focused and include tests or reproducible steps for changes that affect core algorithms or scripts.

Notes & Tips
------------
- Many scripts use relative paths computed from the script location; run them from the repository or ensure working directories are correct.
- When using pretrained .pkl models (joblib), ensure compatible scikit-learn versions.
- For more reproducible TFLite quantization, provide representative datasets during conversion (see conversion scripts).
- If you see errors with MediaPipe installation on Pi or ARM devices, refer to platform-specific build instructions for MediaPipe or use prebuilt wheels.

License
-------
See the LICENSE file in the repository for license details.

Acknowledgments
---------------
- MediaPipe for landmark detection
- TensorFlow and scikit-learn for training and inference primitives
>>>>>>> 87e1c37b6eb491e69818f36db0a134375f0d6285
