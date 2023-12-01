import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import matplotlib.pyplot as plt
import librosa.display

SAMPLE_RATE = 44100
N_MELS = 128
max_time_steps = 109
# Load your audio classification model
model = load_model("audio_classifier18.h5")

# Function to preprocess audio
def preprocess_audio(audio, target_sr=44100, target_shape=(128, 109)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram

# Function to make predictions
def predict(processed_data):
    # Make sure there are no non-finite values in the processed_data
    if not np.all(np.isfinite(processed_data)):
        st.error("Non-finite values detected in the processed audio data. Unable to make predictions.")
        return None

    # Make predictions using your model
    prediction = model.predict(processed_data.reshape(1, 128, 109, 1))

    # Assuming binary classification with softmax output
    class_labels = ["IT IS A FAKE VOICE", "IT IS A REAL VOICE"]
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]

    return predicted_class

# Streamlit app
st.title("Fake AI-generated Audio Detection")
st.title("Record or Upload Audio")

# Record audio using audio_recorder_streamlit
audio_bytes = audio_recorder()

# Option to upload an audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if audio_bytes:
    # Display the recording button
    st.audio(audio_bytes, format='audio/wav', sample_rate=SAMPLE_RATE)

    # Convert audio_bytes to NumPy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Convert to float32 and normalize to the range [-1, 1]
    audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

    # Check for non-finite values
    if not np.all(np.isfinite(audio_np)):
        st.error("Non-finite values detected in the recorded audio. Unable to proceed.")
    else:
        # Preprocess the recorded audio
        processed_data = preprocess_audio(audio_np)

        # Display file details
        st.write("Audio details:")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Duration: {len(audio_np) / SAMPLE_RATE:.2f} seconds")

        # Plot the waveform
        st.subheader("Waveform:")
        fig_waveform, ax_waveform = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(audio_np, sr=SAMPLE_RATE, ax=ax_waveform)
        st.pyplot(fig_waveform)

        # Plot the spectrogram
        st.subheader("Spectrogram:")
        fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(processed_data[:, :], sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax_spec)
        colorbar = plt.colorbar(format='%+2.0f dB', ax=ax_spec, mappable=img)
        st.pyplot(fig_spec)

        # Add a Predict button for recording
        if st.button("Predict (Recording)"):
            # Make predictions
            predicted_class = predict(processed_data)

            # Display predictions
            if predicted_class is not None:
                st.write("Prediction:", predicted_class)

if uploaded_file:
    # Display uploaded file details
    st.audio(uploaded_file, format='audio/wav')
    
    # Preprocess the uploaded audio file
    audio_file = librosa.load(uploaded_file, sr=SAMPLE_RATE)[0]
    processed_data_file = preprocess_audio(audio_file)

    if processed_data_file is not None:
        # Display file details
        st.write("File details:")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Duration: {len(audio_file) / SAMPLE_RATE:.2f} seconds")

        # Plot the waveform
        st.subheader("Waveform:")
        fig_waveform_file, ax_waveform_file = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(audio_file, sr=SAMPLE_RATE, ax=ax_waveform_file)
        st.pyplot(fig_waveform_file)

        # Plot the spectrogram
        st.subheader("Spectrogram:")
        fig_spec_file, ax_spec_file = plt.subplots(figsize=(12, 4))
        img_file = librosa.display.specshow(processed_data_file[:, :], sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax_spec_file)
        colorbar_file = plt.colorbar(format='%+2.0f dB', ax=ax_spec_file, mappable=img_file)
        st.pyplot(fig_spec_file)

        # Add a Predict button for uploaded file
        if st.button("Predict (Uploaded File)"):
            # Make predictions
            predicted_class_file = predict(processed_data_file)

            # Display predictions
            st.write("Prediction:", predicted_class_file)