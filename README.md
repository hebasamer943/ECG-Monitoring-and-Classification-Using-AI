# ECG Monitoring and Classification Using AI

A smart desktop-based ECG monitoring system that supports real-time ECG signal acquisition, visualization, and AI-based classification.

The system can receive ECG data from multiple sources, including hardware signals from ESP32 with AD8232, CSV ECG records, and ECG image input. It displays the ECG waveform, estimates heart rate, evaluates signal quality, and classifies ECG beats as Normal or Abnormal using deep learning models.

---

## Project Overview

This project was developed as a graduation project in Artificial Intelligence and Data Science.

The main goal is to build a practical ECG monitoring system that combines:

- Real-time ECG signal acquisition
- ECG waveform visualization
- Signal quality evaluation
- Beat-level AI classification
- Post-session ECG analysis
- AI-generated medical-style summary report

---

## Key Features

- Real-time ECG monitoring from ESP32 + AD8232 hardware
- CSV-based ECG signal testing
- ECG image input support
- Normal / Abnormal ECG classification
- Heart rate estimation
- Signal quality indicator
- Interactive desktop dashboard
- Session-based analysis
- AI-generated report summary
- Exportable final report

---

## Technologies Used

- Python
- PySide6
- PyQtGraph
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- ESP32
- AD8232 ECG Sensor
- Machine Learning
- Deep Learning

---

## AI Models

The system uses deep learning models for ECG classification.

The real-time monitoring path focuses on binary classification:

- Normal ECG
- Abnormal ECG

Additional post-session analysis can be used for deeper arrhythmia-related evaluation.

---

## Datasets

The project uses ECG datasets for model training and evaluation, including:

- PTB Diagnostic ECG Database
- MIT-BIH Arrhythmia Dataset

---

## Hardware Components

- ESP32 Microcontroller
- AD8232 ECG Sensor Module
- ECG Electrodes
- USB Serial Connection

---

## System Workflow

1. ECG signal is collected from hardware, CSV, or image input.
2. The signal is preprocessed and normalized.
3. The ECG waveform is displayed in the desktop interface.
4. Heart rate and signal quality are calculated.
5. The AI model classifies the ECG signal.
6. A final analysis report can be generated after the session.

---

## Project Purpose

This project demonstrates how artificial intelligence can be integrated with biomedical signals to support ECG monitoring and early abnormality detection.

The system is intended for educational and research purposes, not for direct medical diagnosis.

---


## Supervisor

Dr.Alaa Al-Omoush
---

## Disclaimer

This system is developed for academic and research purposes only.  
It is not a certified medical device and should not be used as a replacement for professional medical diagnosis.
