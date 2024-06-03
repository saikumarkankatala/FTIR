# Microplastic Identification Using ATR-FTIR and Machine Learning

## Overview
This repository contains the code and dataset for the research project aimed at identifying microplastics using Attenuated Total Reflection Fourier Transform Infrared Spectroscopy (ATR-FTIR) combined with advanced machine learning techniques. Our approach transitions from a basic 1D Multi-Layer Perceptron (MLP) model to more sophisticated 1D Convolutional Neural Network (CNN) architectures like AlexNet and VGGNet.

## Repository Structure
- `CreateDataset`: Scripts and utilities to create datasets from raw ATR-FTIR spectra.
- `Datasets`: Contains the preprocessed datasets used in our machine learning models.
- `GetDataFromVids`: Tools to extract data from video files, potentially useful for gathering additional data.
- `MachineLearning`: Contains all machine learning models, training scripts, and evaluation metrics.
- `Spring2024UGR_PosterPrototype_revision5.pptx`: Presentation slides summarizing the research findings.

## Methodology
The methodology employed involves several steps:
1. **Data Collection**: Collection of ATR-FTIR spectra of various microplastics.
2. **Data Preprocessing**: Conversion of spectra into formats suitable for machine learning, including normalization and feature extraction.
3. **Model Training**: Using neural network architectures to learn from the spectra.
4. **Evaluation**: Assessing model performance through accuracy, precision, and recall metrics.

## Results
- Transition from MLP to CNN resulted in significant improvements in classification accuracy.
- AlexNet and VGGNet adaptations for 1D data performed well, with accuracies on validation sets reaching up to 76%.
- Larger datasets significantly enhance model performance, emphasizing the need for expansive spectral libraries.

## Conclusion
Our findings indicate that CNNs, particularly those adapted for 1D data, are highly effective in classifying microplastics from ATR-FTIR spectra. Future work will expand the dataset and refine the models to improve accuracy and reliability in real-world environmental conditions.

## How to Use
1. Clone the repository: git clone https://github.com/saikumarkankatala/FTIR.git
2. Navigate to the `MachineLearning` directory to view training scripts.
3. Datasets can be found under the `Datasets` directory.
