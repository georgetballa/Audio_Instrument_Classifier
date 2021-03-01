import numpy as np
import librosa
from librosa import display
from scipy.io import wavfile
import IPython.display as display
import os
from PIL import Image
import pathlib
import csv
import warnings
import matplotlib.pyplot as plt
import shutil
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras import callbacks
from tensorflow import config
import splitfolders
import numpy as np
from tensorflow.keras import backend as K
import datetime
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
import IPython.display as ipd
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



def add_instrument(instrument, limit, arr):
    j = 0
    
    '''
    Creates Spectrograms for New Instrument Classification from WAV Files
    Saves Files as PNG
    '''
    
    i = instrument
    pathlib.Path(f'/data/instrument_wavs/val/{i}').mkdir(parents=True, exist_ok=True)
    dest = f'/data/instrument_wavs/val/{i}'
    for dirpath, dirnames, filenames in os.walk(r'/data/nsynth-valid/audio/'):
        for filename in filenames:
            if i in filename:
                shutil.move(dirpath + filename, dest)

    pathlib.Path(f'/home/george/Desktop/Audio_Classifier/data/spectrograms/val_set/{i}').mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(f'/data/instrument_wavs/val/{i}'):
        if j < limit:
            j += 1
            soundfile = f'/data/instrument_wavs/va/{i}/{filename}'
            y, sr = librosa.load(soundfile, mono=True, duration=3)
            plt.figure(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=10000)
            librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmax=10000)
            plt.tight_layout()
            plt.axis('off');
            plt.savefig(f'/home/george/Desktop/Audio_Classifier/data/spectrograms/val_set/{i}/{filename[:-3].replace(".", "")}.png',pad_inches = 0,bbox_inches='tight', transparent=True)
            plt.clf()
        else:
            continue
    return
    print(f'finished{i}')

def change_trainable_layers(model, trainable_index):
    '''Set Layers to train for transfer learning'''
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
    '''Create model to transfer learn, add top layers'''
    base_model = VGG19(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        
        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model

def print_model_properties(model, indices = 0):
    'Print layers and boolean of trainability'
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

def eval(idx):
    'evaluate actual vs predicted'
    wrong_bool = y_pred != y_true
    idx_wrong_pred = []
    for idx2,i in enumerate(wrong_bool):

        if i ==True:
            idx_wrong_pred.append(idx2)
    idx_wrong_pred = np.array(idx_wrong_pred)
    wrong_filenames = []
    for i in idx_wrong_pred:
        wrong_filenames.append(filenames[i])
    predicted_at_filenames = y_pred[idx_wrong_pred]
    class_dictionary = val_set.class_indices
    key_list = list(class_dictionary.keys())
    val_list = list(class_dictionary.values())
    position = val_list.index(predicted_at_filenames[idx])
    predicted = key_list[position]
    actual = wrong_filenames[idx]
    return actual, predicted