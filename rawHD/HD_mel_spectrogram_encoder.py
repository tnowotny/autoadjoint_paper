import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import re
from tqdm import trange
import copy
import tarfile
import soundfile as sf
import wget
import random
import pandas as pd

params = {
    "target_steps": 80,
    "num_bands": 80,
    "num_classes": 20,
    "sample_rate": 22050, #16000
    "pre_emphasis": 0.95,
    "frame_length": 0.025,
    "hop_length": 0.01,
    "fft_size": 512,
    "scale_value": 0.00099,
    "save_directory_name": "rawHD_80input",
}

### (1) download and allocate folders for google speech commands dataset ###

# get a directory to download and encode dataset files
directory = os.path.expanduser("~/data")

try:
    os.makedirs(directory)
except:
    pass
os.chdir(directory)

try:
    os.makedirs("rawHD")
except:
    pass
os.chdir("rawHD")

if not os.path.exists("hd_audio.tar.gz"):
    print("downloading dataset")
    wget.download("https://zenkelab.org/datasets/hd_audio.tar.gz")

# unzip to folder
if not os.path.isdir("hd_extracted"):
    # downloading the 35 classes version
    file = tarfile.open("hd_audio.tar.gz")

    file.extractall("./hd_extracted")

    file.close()

os.chdir("hd_extracted")
print("current cwd", os.getcwd())

### (2) sort through dataset to get testing/training ###

# load a list of training audio files
train_files = []
with open("train_filenames.txt", "r") as file:
    for line in file:
        x = line[:-1]
        train_files.append(x)
        
# load a list of testing audio files
test_files = []
with open("test_filenames.txt", "r") as file:
    for line in file:
        x = line[:-1]
        test_files.append(x)

### (3) Mel Spectrogram encoding ###

def to_mel_spectrogram(file_name,
                       params, 
                       display = False):
    
    audio, sr = librosa.load(file_name, sr = params["sample_rate"], mono=True)
    # Apply pre-emphasis filter
    emphasized_audio = np.append(audio[0], 
                                 audio[1:] - params["pre_emphasis"] * audio[:-1])

    # Define frame length and stride in samples
    frame_length = int(sr * params["frame_length"])  # 25ms
    frame_length = min(frame_length, 512)
    hop_length = int(sr * params["hop_length"])  # 10ms

    # Compute the power spectrum using a 512-point FF
    power_spectrum = np.abs(librosa.stft(emphasized_audio, 
                                         n_fft = params["fft_size"], 
                                         hop_length = hop_length, 
                                         win_length = frame_length))**2
    
    # Compute the filter banks with 40 triangular filters
    filter_banks = librosa.filters.mel(n_fft = params["fft_size"], 
                                       sr = sr, 
                                       n_mels = params["num_bands"])

    # Apply the filter banks to the power spectrum
    mel_spec = np.dot(filter_banks, power_spectrum)

    # Crop or pad to 80 steps by repeating the last frame
    current_steps = mel_spec.shape[1]
    if current_steps < params["target_steps"]:
        padding = np.zeros((params["num_bands"], params["target_steps"] - current_steps))
        mel_spec = np.hstack((mel_spec, padding))
    elif current_steps > params["target_steps"]:
        mel_spec = mel_spec[:, :params["target_steps"]]

    # Convert power spectrogram to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if display:
        # Display the filter banks with the 'viridis' colormap
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Filter Banks with Pre-Emphasis Filter (Cropped/Padded to 80 Steps)')
        plt.tight_layout()
        plt.show()
        
        #print(mel_spec_db.shape)
    
    else:
        return mel_spec_db


### (3.5) change directory and Visualise the output of the mel encoding

try:
    os.chdir("audio")
except:
    pass
rnd_val = 10#np.random.randint(0, len(test_files))
test_image = test_files[rnd_val]
to_mel_spectrogram(test_image, params, True)

# Visualise the same input but on soecgram (visual check)
data, samplerate = sf.read(test_image)  
Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)
plt.show()

### (4) loop through audio and convert to Mel Encoding ###

training_x_data = []
training_y_data = []
testing_x_data = []
testing_y_data = []

training_details = pd.DataFrame({'Language': [], 
                                 'Speaker': [], 
                                 'Trial': [], 
                                 'Label': [],
                                 'classification label': []})

testing_details = pd.DataFrame({'Language': [], 
                                 'Speaker': [], 
                                 'Trial': [], 
                                 'Label': [],
                                 'classification label': []})

# save all to a list
for i in trange(len(os.listdir())):
    split_values = re.split("[. _ -]", os.listdir()[i])
    
    if os.listdir()[i] in train_files:
        training_x_data.append(copy.deepcopy(to_mel_spectrogram(os.listdir()[i], params)))
        training_y_data.append(int(split_values[7]) if split_values[1] == "english" else int(split_values[7]) + 10)
        training_details.loc[len(training_details)] = {'Language': split_values[1], 
                                                       'Speaker': int(split_values[3]), 
                                                       'Trial': int(split_values[5]), 
                                                       'Label': int(split_values[7]),
                                                       'classification label': (int(split_values[7]) if split_values[1] == "english" else int(split_values[7]) + 10)}
    else:
        testing_x_data.append(copy.deepcopy(to_mel_spectrogram(os.listdir()[i], params)))
        testing_y_data.append(int(split_values[7]) if split_values[1] == "english" else int(split_values[7]) + 10)
        testing_details.loc[len(testing_details)] = {'Language': split_values[1], 
                                                     'Speaker': int(split_values[3]), 
                                                     'Trial': int(split_values[5]), 
                                                     'Label': int(split_values[7]),
                                                     'classification label': (int(split_values[7]) if split_values[1] == "english" else int(split_values[7]) + 10)}

### (5) Preprocess dataset with scaling and transposing

print(training_details.head())

# save dataset
os.chdir(directory)
# create new directory for raw HD
assert os.path.isdir("rawHD") == True
os.chdir("rawHD")

try:
    os.mkdir(params["save_directory_name"])
except:
    pass
  
os.chdir(params["save_directory_name"])
print("current cwd", os.getcwd())

print("pre processing data...")
# data processing
print(np.asarray(training_x_data).shape)
training_x_data = np.swapaxes(training_x_data, 1, 2) 
testing_x_data = np.swapaxes(testing_x_data, 1, 2) 
print(training_x_data.shape)

# move values into positive
training_x_data -= training_x_data.min()
testing_x_data -= testing_x_data.min()

# scale values 
training_x_data *= params["scale_value"]
testing_x_data = testing_x_data * params["scale_value"]

### (6) save dataset as npy ###

np.save("training_x_data.npy", training_x_data)
np.save("training_y_data.npy", training_y_data)
np.save("testing_y_data.npy", testing_y_data)
np.save("testing_x_data.npy", testing_x_data)

training_details.to_csv('training_details.csv')  
testing_details.to_csv('testing_details.csv')  

print("files saved:")
print(os.listdir())

