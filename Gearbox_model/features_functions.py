import numpy as np
import librosa
from scipy.fft import fft

def extract_basics(data):
    means = []
    stds = []
    maxs = []
    mins = []
    medians = []

    for i in data:
        
        means.append(np.mean(i))
        stds.append(np.std(i))
        maxs.append(np.max(i))
        mins.append(np.min(i))
        medians.append(np.median(i))
    
    return means, stds, maxs, mins, medians 

def extract_basics_split(data):
    means = []
    stds = []
    maxs = []
    mins = []
    medians = []

    for i in data:
    
        splits = np.array_split(i, 50)
    
        for j in splits:
        
            j_fft_v1 = fft(j)
            j_fft = 2.0/(j.shape[0]) * np.abs(j_fft_v1[0:j.shape[0]//2])

            # means.append(np.mean(j))
            # stds.append(np.std(j))
            # maxs.append(np.max(j))
            # mins.append(np.min(j))
            # medians.append(np.median(j))

            means.append(np.mean(j_fft))
            stds.append(np.std(j_fft))
            maxs.append(np.max(j_fft))
            mins.append(np.min(j_fft))
            medians.append(np.median(j_fft))

    return means, stds, maxs, mins, medians


def spec(data):
    spectograms = []

    for i in data:
        x = librosa.stft(i)
        spectograms.append(librosa.amplitude_to_db(abs(x)))
        
    return spectograms


def rms(data):
    rms = []

    for i in data:
        S, phase = librosa.magphase(librosa.stft(i))
        rms.append(librosa.feature.rms(S=S))
        
    return rms

def zcr(data):
    zcr = []

    for i in data:
        zcr.append(librosa.feature.zero_crossing_rate(i))
        
    return zcr


def mfcc(data):
    mfcc = []

    for i in data:
        mfcc.append(librosa.feature.mfcc(y=i, sr=16000))
        
    return mfcc