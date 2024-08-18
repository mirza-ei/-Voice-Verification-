

# Voice Verification with Siamese Network

## Overview

This project implements a voice verification system using a Siamese neural network. The system determines whether two audio samples are from the same speaker or different speakers. It uses features extracted from audio files and a Siamese network model to perform this verification.

## Dependencies

To run this project, you need the following Python libraries:
- `tensorflow`
- `numpy`
- `librosa`
- `gradio`

You can install these libraries using pip:

```bash
pip install tensorflow numpy librosa gradio
```

## Code Explanation

### Audio Data Preprocessing

The audio files are processed to extract Short-Time Fourier Transform (STFT) features. These features are used to create pairs of audio samples for training and testing the Siamese network.

### Model Definition

The Siamese network model is implemented using GRU cells with batch normalization and dense layers. The model compares the features from two audio samples using cosine similarity.

### Training and Evaluation

The model is trained over multiple epochs using the training data, and its performance is evaluated on the test data. The training process involves optimizing the model using a loss function and an optimizer.

### Gradio Interface

A Gradio interface is used to create a web application for testing the voice verification system. This interface allows you to upload audio files and see the results of the verification in real-time.

## Usage

1. **Prepare your audio files**: Ensure that you have two audio files you want to compare.
2. **Run the script**: Execute the script to start the Gradio web interface.
3. **Upload audio files**: In the web interface, upload the reference and test audio files.
4. **View results**: The system will indicate whether the speakers are the same or different based on the model's prediction.

## Troubleshooting

- **Error: One or both audio files are not uploaded properly**: Make sure that both audio files are uploaded correctly.
- **Error processing files**: Check the console for error details to troubleshoot issues related to file processing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

