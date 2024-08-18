

# Siamese Network for Audio Similarity

This project uses a Siamese network with GRU (Gated Recurrent Unit) cells to compare audio files based on their Short-Time Fourier Transform (STFT) representations. The model is designed to classify pairs of audio samples as either similar or dissimilar.

## Overview

1. **Data Loading**: Loads training and test audio data from Google Drive.
2. **Preprocessing**: Converts audio data to STFT and computes the absolute value.
3. **Dataset Generation**: Creates pairs of audio samples for training and testing, both positive (same speaker) and negative (different speakers).
4. **Model Definition**: Uses a Siamese network architecture with GRU cells and batch normalization.
5. **Training**: Trains the model and evaluates performance based on accuracy and loss.

## Usage

1. **Mount Google Drive**: The script requires access to Google Drive to load data. Ensure that the data files `hw4_trs.pkl` and `hw4_tes.pkl` are available in the specified paths on Google Drive.

2. **Run the Script**: Execute the script in a Python environment that supports TensorFlow 1.x.

```sh
python your_script.py
```

3. **Monitoring**: The script will print the test loss and accuracy after each epoch. The training will stop once the accuracy exceeds 70%.

## Notes

- **Data Files**: The `.pkl` files should contain preprocessed audio data compatible with the script.
- **Model Parameters**: You can adjust the model parameters such as hidden units, dropout rate, learning rate, and number of epochs in the script.
- **TensorFlow Version**: This code is compatible with TensorFlow 1.x. For TensorFlow 2.x, code modifications will be needed.

