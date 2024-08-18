import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import gradio as gr
import librosa

@tf.keras.utils.register_keras_serializable()
class EuclideanDistanceLayer(Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True))
    
    def get_config(self):
        return super(EuclideanDistanceLayer, self).get_config()

# Load the pre-trained model
model = tf.keras.models.load_model('siamese_net_final.keras', custom_objects={'EuclideanDistanceLayer': EuclideanDistanceLayer})

def preprocess_audio_data(file_path):
    """
    Preprocess the audio file to extract MFCC features.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Processed audio features.
    """
    audio_data, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=513)
    mfccs_flat = np.mean(mfccs, axis=1)
    return mfccs_flat.reshape(1, -1)

def verify_voices(reference_audio, test_audio):
    """
    Verify if two audio files are from the same speaker or not.

    Args:
        reference_audio (str): Path to the reference audio file.
        test_audio (str): Path to the test audio file.

    Returns:
        str: Result of the verification.
    """
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

# Create Gradio interface
iface = gr.Interface(
    fn=verify_voices,
    inputs=[gr.Audio(type="filepath"), gr.Audio(type="filepath")],
    outputs="text",
    live=True
)

# Launch the interface
iface.launch()
