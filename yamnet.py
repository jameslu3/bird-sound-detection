import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
import scipy.signal
from scipy.io import wavfile
import tensorflow_hub as hub

# Load the YAMNet model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet = hub.load(yamnet_model_handle)

# Preprocess audio 
def preprocess_audio(audio, sample_rate):
    target_rate = 16000
    if sample_rate != target_rate:
        audio = tf.audio.resample(audio, rate_in=sample_rate, rate_out=target_rate)
    return audio

# Get embeddings from YAMNet
def extract_yamnet_embedding(audio, sample_rate):
    audio = preprocess_audio(audio, sample_rate)
    scores, embeddings, spectrogram = yamnet(audio)
    return embeddings.numpy()

# Below code copied from TensorFlow Hub documentation https://www.tensorflow.org/hub/tutorials/yamnet
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = yamnet_model_handle.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform