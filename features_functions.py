import numpy as np
import librosa


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