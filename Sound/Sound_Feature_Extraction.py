import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import csv
import keras
import wave, os, glob
class Sound_Feature_Extraction():
    def create_Features(self):
        header = 'time chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
                    header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        file = open('sound_data.csv', 'a', newline='')
        with file:
                writer = csv.writer(file)
                writer.writerow(header)
        time=0
        label=0 #for not cheation values label=0, otherwise label=1   
            
        path = 'splited' #folder where all the splitted audios are saved
        for filename in glob.glob(os.path.join(path, '*.wav')):
                    time+=1
                    label=0
                    y, sr = librosa.load(filename, mono=True, duration=30)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    to_append = f' {time} {np.mean(chroma_stft)}  {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}' 

                    for e in mfcc:
                        to_append+=f' {np.mean(e)}'
                    to_append+=f' {label}'
                        
                            
                    file = open('sound_data.csv', 'a', newline='')
                    with file:
                            writer = csv.writer(file)
                            writer.writerow(to_append.split())