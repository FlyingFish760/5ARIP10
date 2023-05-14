
#Import libaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile 
import os

def loading_data(directory, section, voltage):
    
    # create empty lists to store the data
    anom_files = []
    anom_sr = []
    anom_data = []
    anom_data_fft = []

    norm_files = []
    norm_sr = []
    norm_data = []
    norm_data_fft = []

    # make a list of all the files in the training data folder that contain section and voltage in the name
    for i in os.listdir(directory):
        if 'anomaly' in i and section in i and voltage in i:
            anom_files.append(i)
            # samplerate, data = wavfile.read(directory + '\\' + i) # for windows
            samplerate, data = wavfile.read(directory + '/' + i) # for macos
            anom_sr.append(samplerate)
            anom_data.append(data)

            #also store the fft data
            data_fft_v1 = fft(data)
            data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
            anom_data_fft.append(data_fft)

        if 'normal' in i and section in i and voltage in i:
            norm_files.append(i)
            # samplerate, data = wavfile.read(directory + '\\' + i) # for windows
            samplerate, data = wavfile.read(directory + '/' + i) # for macos
            norm_sr.append(samplerate)
            norm_data.append(data)

            #Also store the fft data
            data_fft_v1 = fft(data)
            data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
            norm_data_fft.append(data_fft)
    
    return anom_files, anom_sr, anom_data, anom_data_fft, norm_files, norm_sr, norm_data, norm_data_fft

def split_extract_features(data, fft_bool=False):
    means = []
    stds = []
    maxs = []
    mins = []
    medians = []

    for i in data:
        
        splits = np.array_split(i, 50)
        
        for j in splits:
            
            if fft_bool:
                j_fft = fft(j)
                split = 2.0/(j.shape[0]) * np.abs(j_fft[0:j.shape[0]//2])
            else:
                split = j
            
            means.append(np.mean(split))
            stds.append(np.std(split))
            maxs.append(np.max(split))
            mins.append(np.min(split))
            medians.append(np.median(split))
    
    return means, stds, maxs, mins, medians

def plot_basics(anom_means, anom_stds, anom_maxs, anom_mins, anom_medians, norm_means, norm_stds, norm_maxs, norm_mins, norm_medians):
    
    fig, ax = plt.subplots(5, 2,figsize=(7,15), tight_layout=True)

    ax[0,0].scatter(norm_means, norm_stds, alpha=0.3)
    ax[0,0].scatter(anom_means, anom_stds, alpha=0.3)
    ax[0,0].grid()
    ax[0,0].set_xlabel(r'mean', fontsize=10)
    ax[0,0].set_ylabel(r'std', fontsize=10)
    ax[0,0].set_title(r'Cluster option 1', fontsize=14)

    ax[0,1].scatter(norm_means, norm_maxs, alpha=0.3)
    ax[0,1].scatter(anom_means, anom_maxs, alpha=0.3)
    ax[0,1].grid()
    ax[0,1].set_xlabel(r'mean', fontsize=10)
    ax[0,1].set_ylabel(r'max', fontsize=10)
    ax[0,1].set_title(r'Cluster option 2', fontsize=14)

    ax[1,0].scatter(norm_means, norm_mins, alpha=0.3)
    ax[1,0].scatter(anom_means, anom_mins, alpha=0.3)
    ax[1,0].grid()
    ax[1,0].set_xlabel(r'mean', fontsize=10)
    ax[1,0].set_ylabel(r'min', fontsize=10)
    ax[1,0].set_title(r'Cluster option 3', fontsize=14)

    ax[1,1].scatter(norm_means, norm_medians, alpha=0.3)
    ax[1,1].scatter(anom_means, anom_medians, alpha=0.3)
    ax[1,1].grid()
    ax[1,1].set_xlabel(r'mean', fontsize=10)
    ax[1,1].set_ylabel(r'median', fontsize=10)
    ax[1,1].set_title(r'Cluster option 4', fontsize=14)

    ax[2,0].scatter(norm_stds, norm_maxs, alpha=0.3)
    ax[2,0].scatter(anom_stds, anom_maxs, alpha=0.3)
    ax[2,0].grid()
    ax[2,0].set_xlabel(r'std', fontsize=10)
    ax[2,0].set_ylabel(r'max', fontsize=10)
    ax[2,0].set_title(r'Cluster option 5', fontsize=14)

    ax[2,1].scatter(norm_stds, norm_mins, alpha=0.3)
    ax[2,1].scatter(anom_stds, anom_mins, alpha=0.3)
    ax[2,1].grid()
    ax[2,1].set_xlabel(r'std', fontsize=10)
    ax[2,1].set_ylabel(r'min', fontsize=10)
    ax[2,1].set_title(r'Cluster option 6', fontsize=14)

    ax[3,0].scatter(norm_stds, norm_medians, alpha=0.3)
    ax[3,0].scatter(anom_stds, anom_medians, alpha=0.3)
    ax[3,0].grid()
    ax[3,0].set_xlabel(r'std', fontsize=10)
    ax[3,0].set_ylabel(r'median', fontsize=10)
    ax[3,0].set_title(r'Cluster option 7', fontsize=14)

    ax[3,1].scatter(norm_maxs, norm_mins, alpha=0.3)
    ax[3,1].scatter(anom_maxs, anom_mins, alpha=0.3)
    ax[3,1].grid()
    ax[3,1].set_xlabel(r'max', fontsize=10)
    ax[3,1].set_ylabel(r'min', fontsize=10)
    ax[3,1].set_title(r'Cluster option 8', fontsize=14)

    ax[4,0].scatter(norm_maxs, norm_medians, alpha=0.3)
    ax[4,0].scatter(anom_maxs, anom_medians, alpha=0.3)
    ax[4,0].grid()
    ax[4,0].set_xlabel(r'max', fontsize=10)
    ax[4,0].set_ylabel(r'median', fontsize=10)
    ax[4,0].set_title(r'Cluster option 9', fontsize=14)

    ax[4,1].scatter(norm_mins, norm_medians, alpha=0.3)
    ax[4,1].scatter(anom_mins, anom_medians, alpha=0.3)
    ax[4,1].grid()
    ax[4,1].set_xlabel(r'min', fontsize=10)
    ax[4,1].set_ylabel(r'max', fontsize=10)
    ax[4,1].set_title(r'Cluster option 10', fontsize=14)
    plt.legend(['Normal','Anomalous'])
    
    return plt.show()