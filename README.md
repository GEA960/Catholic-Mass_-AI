# Catholic-Mass_-AI
An AI that will detect what a parish priest was saying on the mass and will play a certain music on every part of the mass if reached certain sound parameters.

The goal of this project is to aid the choir in any parishes if ever no one or the person who were assigned to that job is not available and can't facilitate the music accompaniment of the mass. The AI will be the one who will play the music in minus one or with voice. It will depend on the availability of the choir.


PROGRAMMERS!

Let's create an AI na magpa play ng music whether minus one or may voice once reached ang some parameters. I just have an idea to give support sa choir once hindi po available ang tagatugtog or taga play ng accompaniment. 

Parameters:
     The AI will play a mass song depende sa sinabi na words ng pari and ng commentator sa misa. So real-time processing po.
     So every cue from the parish priests' word, papatugtugin ng AI ang mga naka assign na songs line up for that Mass. 

P.S. aaralin ko pa po ito from scratch sa python. So kung may gusto po makipag collaborate, please connect with me. Thank you!

 Collaborate with me here:
https://github.com/GEA960/Catholic-Mass_-AI.git

Instructions: 

To create a Python program that outputs a music when it hears a specific external music, we will need to use a library that can listen to the external sound input and a library that can output a music file. We can use PyAudio for listening to the external sound input and pygame for outputting the music file.

Here's an example code that listens to an external music input and outputs a music file when it hears a specific external music:

python
Copy code
import pyaudio
import wave
import numpy as np
import pygame

# Define the audio input parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3

# Define the external music file path
music_path = "external_music.wav"

# Define the music to be played when the specific external music is detected
output_music_path = "output_music.mp3"

# Open the output music file
pygame.mixer.init()
pygame.mixer.music.load(output_music_path)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the audio stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

# Loop until the specific external music is detected
while True:
    # Read audio data from the stream
    data = stream.read(CHUNK)
    # Convert audio data to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Check if the specific external music is detected
    if np.any(audio_data > 5000):
        # Play the output music file
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

# Close the audio stream and PyAudio
stream.stop_stream()
stream.close()
audio.terminate()
In this code, we define the audio input parameters and the external music file path. We also define the music to be played when the specific external music is detected and the output music file path. We open the output music file using pygame.mixer and initialize PyAudio. We then open the audio stream and loop until the specific external music is detected. We read audio data from the stream, convert it to a numpy array, and check if the specific external music is detected by checking if any of the audio data values are greater than 5000. If the specific external music is detected, we play the output music file using pygame.mixer.music. Finally, we close the audio stream and PyAudio.

Note: This code is just an example and may need to be modified depending on the specific requirements of the project.



To write a program that can recognize a specific external music and play a corresponding music, you would need to use audio processing libraries and machine learning techniques. Here's a high-level overview of the steps involved:

Data Collection: Collect audio samples of the music you want to recognize and play. You'll need a few different samples of the same music to train your machine learning model.

Preprocessing: Before feeding the audio data into your machine learning model, you'll need to preprocess the audio signals to extract relevant features that the model can use. Some examples of audio features include pitch, tempo, and spectral information.

Training a Model: Using the preprocessed audio data, you'll train a machine learning model to recognize the music you want to play. There are many different machine learning algorithms you could use for this task, such as a neural network or a decision tree.

Real-Time Recognition: Once you have a trained model, you can use it to recognize the external music in real-time. You'll need to use an audio processing library to capture the incoming audio signal, and then pass that signal through your machine learning model to make a prediction.

Playing Music: If the recognized music matches the music you want to play, you can use an audio playback library to play the corresponding music.

Here's some example code to get you started with the audio processing and machine learning parts:

python
Copy code
import librosa
import numpy as np
import tensorflow as tf

# Load audio file
audio, sr = librosa.load('external_music.wav', sr=None)

# Extract audio features
mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
spectral_centroid = librosa.feature.spectral_centroid(audio, sr=sr)
features = np.concatenate([mfccs, spectral_centroid], axis=0).T

# Load trained model
model = tf.keras.models.load_model('music_recognition_model.h5')

# Define labels for recognized music classes
labels = ['music_1', 'music_2', 'music_3']

# Start real-time recognition loop
while True:
    # Capture incoming audio signal
    audio_signal = capture_audio_signal()

    # Extract features from incoming audio
    mfccs = librosa.feature.mfcc(audio_signal, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(audio_signal, sr=sr)
    incoming_features = np.concatenate([mfccs, spectral_centroid], axis=0).T

    # Make prediction using trained model
    prediction = model.predict(incoming_features)
    predicted_class = labels[np.argmax(prediction)]

    # If recognized music matches target music, play corresponding music
    if predicted_class == 'target_music':
        play_music('target_music.mp3')
Note that this is just an example and you'll need to modify it to suit your specific needs. You'll also need to research and experiment with different audio processing and machine learning techniques to get the best results.









