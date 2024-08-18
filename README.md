

# Voice Verification with Siamese Network

## Overview

This project implements a voice verification system using a Siamese neural network. The system is designed to determine whether two audio samples are from the same speaker or different speakers. It uses a pre-trained Siamese network model and leverages audio features extracted using Mel-Frequency Cepstral Coefficients (MFCCs).

## Dependencies

The project requires the following Python libraries:
- `tensorflow`
- `numpy`
- `gradio`
- `librosa`

You can install these libraries using pip:

```bash
pip install tensorflow numpy gradio librosa
```

## Code Explanation

### EuclideanDistanceLayer

A custom Keras layer that computes the Euclidean distance between two input vectors.

```python
@tf.keras.utils.register_keras_serializable()
class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True))
    
    def get_config(self):
        return super(EuclideanDistanceLayer, self).get_config()
```

### Model Loading

The pre-trained Siamese network model is loaded using the `tf.keras.models.load_model` method.

```python
model = tf.keras.models.load_model('siamese_net_final.keras', custom_objects={'EuclideanDistanceLayer': EuclideanDistanceLayer})
```

### Audio Data Preprocessing

The `preprocess_audio_data` function processes audio files by extracting MFCC features.

```python
def preprocess_audio_data(file_path):
    audio_data, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=513)
    mfccs_flat = np.mean(mfccs, axis=1)
    return mfccs_flat.reshape(1, -1)
```

### Voice Verification

The `verify_voices` function compares two audio files to determine if they are from the same speaker.

```python
def verify_voices(reference_audio, test_audio):
    if reference_audio is None or test_audio is None:
        return "Error: One or both audio files are not uploaded properly."
    try:
        reference_features = preprocess_audio_data(reference_audio)
        test_features = preprocess_audio_data(test_audio)
        prediction = model.predict([reference_features, test_features])
        return "Same Speaker" if prediction > 0.5 else "Different Speakers"
    except Exception as e:
        print(f"Error: {e}")
        return "Error processing files"
```

### Gradio Interface

The Gradio interface is used to create a web application for testing the voice verification system.

```python
iface = gr.Interface(
    fn=verify_voices,
    inputs=[gr.Audio(type="filepath"), gr.Audio(type="filepath")],
    outputs="text",
    live=True
)
iface.launch()
```

## Usage

1. **Prepare your audio files**: Ensure that you have two audio files you want to compare.
2. **Run the script**: Execute the script to start the Gradio web interface.
3. **Upload audio files**: In the web interface, upload the reference and test audio files.
4. **View results**: The system will indicate whether the speakers are the same or different based on the model's prediction.

## Troubleshooting

- **Error: One or both audio files are not uploaded properly**: Ensure that you have uploaded both audio files correctly.
- **Error processing files**: Check the console for error details to troubleshoot issues related to file processing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

