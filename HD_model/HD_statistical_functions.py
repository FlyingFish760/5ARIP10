
#Import libaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile 
import os
import csv


def load_acc_data(dir_data, locat, date, fs, secs):
    
    dir_acc = os.path.join(dir_data, locat, 'Accelerometer', date)

    acc_files = []
    acc_time = []
    acc_data = []
    acc_data_fft = []


    for i in os.listdir(dir_acc):    
        if f"fs{fs}" in i and f"secs{secs}" in i:
            acc_files.append(i) 

            with open(dir_acc + '/' + i, 'r') as file:
                csv_data = np.asarray(list(csv.reader(file)))
                
            #samplerate, data = wavfile.read(dir_acc + '\\' + i) # for windows
            acc_time.append(csv_data[1:,0])

            data = csv_data[1:,1:4]
            acc_data.append(data)

            #also store the fft data
            data_fft_v1 = fft(data)
            data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
            acc_data_fft.append(data_fft)
    
    acc_files = np.array(acc_files)
    acc_time = np.array(acc_time)
    acc_data = np.array(acc_data)
    acc_data_fft = np.array(acc_data_fft)


    return  acc_files, acc_time, acc_data, acc_data_fft

def load_mic_data(dir_data, locat, date, fs, secs):
    
    dir_mic = os.path.join(dir_data, locat, 'Microphone', date)

    mic_files = []
    mic_sr = []
    mic_data = []
    mic_data_fft = []

    for i in os.listdir(dir_mic):   
        if f"fs{fs}" in i and f"secs{secs}" in i:
            mic_files.append(i)
            
            #samplerate, data = wavfile.read(dir_acc + '\\' + i) # for windows
            samplerate, data = wavfile.read(dir_mic + '/' + i) # for macos
            mic_sr.append(samplerate)
            mic_data.append(data)

            #also store the fft data
            data_fft_v1 = fft(data)
            data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
            mic_data_fft.append(data_fft)
    
    mic_files = np.array(mic_files)
    mic_sr = np.array(mic_sr)
    mic_data = np.array(mic_data)
    mic_data_fft = np.array(mic_data_fft)

    return  mic_files, mic_sr, mic_data, mic_data_fft


def plot_recording(acc_time,acc_data,mic_sr,mic_data):
    fig, ax = plt.subplots(4,figsize=(7,15), tight_layout=True)

    ax[0].plot(acc_time, acc_data[:,0])
    ax[0].grid()
    ax[0].set_xlabel(r'time [s]', fontsize=10)
    ax[0].set_ylabel(r'x-value acceleration', fontsize=10)
    ax[0].set_title(r'Acceleration data', fontsize=14)

    ax[1].plot(acc_time, acc_data[:,1])
    ax[1].grid()
    ax[1].set_xlabel(r'time [s]', fontsize=10)
    ax[1].set_ylabel(r'y-value acceleration', fontsize=10)
    ax[1].set_title(r'Acceleration data', fontsize=14)

    ax[2].plot(acc_time, acc_data[:,2])
    ax[2].grid()
    ax[2].set_xlabel(r'time [s]', fontsize=10)
    ax[2].set_ylabel(r'y-value acceleration', fontsize=10)
    ax[2].set_title(r'Acceleration data', fontsize=14)

    mic_time = np.linspace(0,int(round(acc_time[-1])),mic_data.shape[0])

    ax[3].plot(mic_time, mic_data[:,0])
    ax[3].plot(mic_time, mic_data[:,1])
    ax[3].grid()
    ax[3].set_xlabel(r'time [s]', fontsize=10)
    ax[3].set_ylabel(r'magnitude microphone', fontsize=10)
    ax[3].set_title(r'Microphone data', fontsize=14)

    return plt.show()


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