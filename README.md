# BISINDO Sign Language Translator

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
