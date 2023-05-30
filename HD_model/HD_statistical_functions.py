
#Import libaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile 
import os
import csv

def loading_mic_data(directory, date, condition):
    
    if condition == 'working':
        # dir_acc = os.path.join(directory, date, "Accelerometer")
        dir_mic = os.path.join(directory, date, "working_drive\Microphone")
    elif condition == 'faulty': 
        dir_mic = os.path.join(directory, date, "faulty_drive\Microphone")

    acc_files = []
    acc_time = []
    acc_data = []
    acc_data_fft = []

    mic_files = []
    mic_sr = []
    mic_data = []
    mic_data_fft = []

    # for i in os.listdir(dir_acc):
    #     acc_files.append(i)

    #     with open(dir_acc + '/' + i, 'r') as file:
    #         csv_data = np.asarray(list(csv.reader(file)))
            
    #     #samplerate, data = wavfile.read(dir_acc + '\\' + i) # for windows
    #     acc_time.append(csv_data[1:,0])

    #     data = csv_data[1:,1:4]
    #     acc_data.append(data)

    #     #also store the fft data
    #     data_fft_v1 = fft(data)
    #     data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
    #     acc_data_fft.append(data_fft)

    for i in os.listdir(dir_mic):
        mic_files.append(i)
        
        #samplerate, data = wavfile.read(dir_acc + '\\' + i) # for windows
        samplerate, data = wavfile.read(dir_mic + '/' + i) # for macos
        mic_sr.append(samplerate)
        mic_data.append(data)

        #also store the fft data
        data_fft_v1 = fft(data)
        data_fft = 2.0/(data.shape[0]) * np.abs(data_fft_v1[0:data.shape[0]//2])
        mic_data_fft.append(data_fft)
    
    return acc_files, acc_time, acc_data, acc_data_fft, mic_files, mic_sr, mic_data, mic_data_fft



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
    

def split_extract_vibr_features(data, x):
    """ 
    Calculate the statistical features for every x sample points along the last dimension

    return: 
    feature map of vibration features (both in time and frequenct domain) ->
    x_vibr_means,x_vibr_maxs...x_vibr_fft_means,x_vibr_fft_maxs...y...z...
    type: np array
    size: (n_files * n_time_window(fs*secs/x)) * (30 -> 3(x,y,z)*10(5 time domain features + 5 frequency domain features))
    
    """

    # Reshape the data to split them
    reshaped_data = np.reshape(data, (data.shape[0], data.shape[1], -1, x))  
    #calculate fft of data
    reshaped_data_fft = np.abs(np.fft.fft(reshaped_data, axis=3))

    # calculate statistical features of fft data
    # use split function to split the data into 3 directions (x, y, z dirctions)
    vibr_fft_means = np.split(np.mean(reshaped_data_fft, axis=-1), 3, axis=1)
    vibr_fft_maxs = np.split(np.max(reshaped_data_fft, axis=-1), 3, axis=1)
    vibr_fft_mins = np.split(np.min(reshaped_data_fft, axis=-1), 3, axis=1)
    vibr_fft_medians = np.split(np.median(reshaped_data_fft, axis=-1), 3, axis=1)
    vibr_fft_stds = np.split(np.std(reshaped_data_fft, axis=-1), 3, axis=1)

    # use np.squeeze to eliminate the dimension with only one element
    vibr_fft_means = np.squeeze(vibr_fft_means)
    vibr_fft_maxs = np.squeeze(vibr_fft_maxs)
    vibr_fft_mins = np.squeeze(vibr_fft_mins)
    vibr_fft_medians = np.squeeze(vibr_fft_medians)
    vibr_fft_stds = np.squeeze(vibr_fft_stds)
    
    # split data into separate x,y,z directions
    x_vibr_fft_means, y_vibr_fft_means, z_vibr_fft_means = vibr_fft_means
    x_vibr_fft_maxs, y_vibr_fft_maxs, z_vibr_fft_maxs = vibr_fft_maxs
    x_vibr_fft_mins, y_vibr_fft_mins, z_vibr_fft_mins = vibr_fft_mins
    x_vibr_fft_medians, y_vibr_fft_medians, z_vibr_fft_medians = vibr_fft_medians
    x_vibr_fft_stds, y_vibr_fft_stds, z_vibr_fft_stds = vibr_fft_stds


    # calculate statistical features of original data
    vibr_means = np.split(np.mean(reshaped_data, axis=-1), 3, axis=1)
    vibr_maxs = np.split(np.max(reshaped_data, axis=-1), 3, axis=1)
    vibr_mins = np.split(np.min(reshaped_data, axis=-1), 3, axis=1)
    vibr_medians = np.split(np.median(reshaped_data, axis=-1), 3, axis=1)
    vibr_stds = np.split(np.std(reshaped_data, axis=-1), 3, axis=1)

    # use np.squeeze to eliminate the dimension with only one element
    vibr_means = np.squeeze(vibr_means)
    vibr_maxs = np.squeeze(vibr_maxs)
    vibr_mins = np.squeeze(vibr_mins)
    vibr_medians = np.squeeze(vibr_medians)
    vibr_stds = np.squeeze(vibr_stds)
    
    # split data into separate x,y,z directions
    x_vibr_means, y_vibr_means, z_vibr_means = vibr_means
    x_vibr_maxs, y_vibr_maxs, z_vibr_maxs = vibr_maxs
    x_vibr_mins, y_vibr_mins, z_vibr_mins = vibr_mins
    x_vibr_medians, y_vibr_medians, z_vibr_medians = vibr_medians
    x_vibr_stds, y_vibr_stds, z_vibr_stds = vibr_stds

    # integrade data to a high dimensional feature matrix: size: (n_speeds * n_time_window) * (3(x, y,z) * 10(5 original 
    # features + 5 fft features)); 
    # sequence of features: x_vibr_means,x_vibr_maxs...x_vibr_fft_means,x_vibr_fft_maxs...y...z...
    vibr_feats = []
    for i in [x_vibr_means, x_vibr_maxs, x_vibr_mins, x_vibr_medians, x_vibr_stds, 
              x_vibr_fft_means, x_vibr_fft_maxs, x_vibr_fft_mins, x_vibr_fft_medians, x_vibr_fft_stds,
              y_vibr_means, y_vibr_maxs, y_vibr_mins, y_vibr_medians, y_vibr_stds, 
              y_vibr_fft_means, y_vibr_fft_maxs, y_vibr_fft_mins, y_vibr_fft_medians, y_vibr_fft_stds,
              z_vibr_means, z_vibr_maxs, z_vibr_mins, z_vibr_medians, z_vibr_stds, 
              z_vibr_fft_means, z_vibr_fft_maxs, z_vibr_fft_mins, z_vibr_fft_medians, z_vibr_fft_stds]:
        i = i.flatten()
        vibr_feats.append(i)

    vibr_feats = np.array(vibr_feats).T
    
    return vibr_feats

def split_extract_mic_features(data, x):
    """ 
    Calculate the statistical features for every x sample points
    
    return: 
    feature map of microphone features (both in time and frequenct domain) ->
    mic_means,mic_maxs...mic_fft_means,mic_fft_maxs...
    type: np array
    size: (n_files * n_time_window:fs*secs/x) * (10: 5 time domain features + 5 frequency domain features)
    """
    # only keep the left stereo, since the difference of left and right stereos is very small:
    # shape:[11*1*2646000*2]-> shape:[11*1*2646000*2]
    data = data[:, :, :, 0]

    # eliminate dimensions of only one element
    # shape:[11*1*2646000*1]-> shape:[11*2646000]
    data = np.squeeze(data)

    # Reshape the data to split them every x sample points
    # shape:[11*2646000]-> shape:[11*(2646000/x)*x]
    reshaped_data = np.reshape(data, (data.shape[0], -1, x)) 

    #calculate fft of data
    reshaped_data_fft = np.abs(np.fft.fft(reshaped_data, axis=2))

    # calculate statistical features of fft data
    # shape:[11*(2646000/x)*x]-> shape:[11*(2646000/x)]
    mic_fft_means = np.mean(reshaped_data_fft, axis=-1)
    mic_fft_maxs = np.max(reshaped_data_fft, axis=-1)
    mic_fft_mins = np.min(reshaped_data_fft, axis=-1)
    mic_fft_medians = np.median(reshaped_data_fft, axis=-1)
    mic_fft_stds = np.std(reshaped_data_fft, axis=-1)
    


    # calculate statistical features of original data
    # shape:[11*(2646000/x)*x]-> shape:[11*(2646000/x)]
    mic_means = np.mean(reshaped_data, axis=-1)
    mic_maxs = np.max(reshaped_data, axis=-1)
    mic_mins = np.min(reshaped_data, axis=-1)
    mic_medians = np.median(reshaped_data, axis=-1)
    mic_stds = np.std(reshaped_data, axis=-1)

    # integrade data to a high dimensional feature matrix: size: (n_speeds * n_time_window) * (10(5 original 
    # features + 5 fft features)); 
    # sequence of features: mic_means,mic_maxs...mic_fft_means,mic_fft_maxs...
    mic_feats = []
    for i in [mic_means, mic_maxs, mic_mins, mic_medians, mic_stds, 
              mic_fft_means, mic_fft_maxs, mic_fft_mins, mic_fft_medians, mic_fft_stds]:
        i = i.flatten()
        mic_feats.append(i)

    mic_feats = np.array(mic_feats).T
    
    return mic_feats

def extract_time_features(data, x):
    """ 
    extract statistical features of data in time domain for every x sample points

    return:
    feature map (type: np array; 
    size: (n_files * n_time_window: fs*secs/x) * (15 features))
    
    """
    # Reshape the data to split them every x sample points
    reshaped_data = np.resize(data, (data.shape[0], int(data.shape[1]/x), x))  

    def mean(array):
        mean = np.mean(array)
        return mean
    
    def max(array):
        max = np.max(array)
        return max
    
    def Root_Mean_Square(array):
        RMS = np.sqrt(np.mean(np.square(array)))
        return RMS
    def Square_Root_Mean(array):
        SRM = np.square(np.mean(np.sqrt(np.abs(array))))
        return SRM
    def Standard_Deviation(array):
        SD = np.std(array)
        return SD
    def Variance(array):
        vari = np.var(array)
        return vari
    def Form_Factor_with_RMS(array):
        RMS = np.sqrt(np.mean(np.square(array)))
        FF_RMS = RMS/(np.mean(np.abs(array)))
        return FF_RMS
    def Form_Factor_with_SRM(array):
        SRM = np.square(np.mean(np.sqrt(np.abs(array))))
        FF_SRM = SRM/(np.mean(np.abs(array)))
        return FF_SRM
    def Crest_Factor(array):
        max = np.max(array)
        RMS = np.sqrt(np.mean(np.square(array)))
        CF = max/RMS
        return CF
    def Latitude_Factor(array):
        max = np.max(array)
        SRM = np.square(np.mean(np.sqrt(np.abs(array))))
        LF = max/SRM
        return LF
    def Impulse_Factor(array):
        max = np.max(array)
        IF = max/(np.mean(np.abs(array)))
        return IF
    def Skewness(array):
        Exp = np.mean(np.power((array - np.mean(array)), 3))
        SD = np.std(array)
        sk = Exp/np.power(SD, 3)
        return sk
    def Kurtosis(array):
        Exp = np.mean(np.power((array - np.mean(array)), 4))
        SD = np.std(array)
        kur = Exp/np.power(SD, 4)
        return kur
    def Moment_5th(array):
        Exp = np.mean(np.power((array - np.mean(array)), 5))
        SD = np.std(array)
        M5 = Exp/np.power(SD, 5)
        return M5
    def Moment_6th(array):
        Exp = np.mean(np.power((array - np.mean(array)), 6))
        SD = np.std(array)
        M6 = Exp/np.power(SD, 6)
        return M6
    
    feature_map = []
    orig_data = reshaped_data
    functions = [mean, max, Root_Mean_Square, Square_Root_Mean, Standard_Deviation, Variance, Form_Factor_with_RMS, 
                   Form_Factor_with_SRM, Crest_Factor, Latitude_Factor, Impulse_Factor, Skewness, Kurtosis, Moment_5th,
                   Moment_6th]
    for function in functions:
        array_feat  = np.zeros((reshaped_data.shape[0], reshaped_data.shape[1]))
        for i in range(orig_data.shape[0]):
            for j in range(orig_data.shape[1]):
                array_feat[i, j] = function(orig_data[i, j, :])   
        array_feat = array_feat.flatten()    
        feature_map.append(array_feat)

    feature_map = np.array(feature_map).T
    return feature_map

def extract_fft_features(data, x):
    """ 
    extract statistical features of fft reshaped_data for every x sample points

    return:
    fft feature map (type: np array; 
    size: (n_files * n_time_window:fs*secs/x) * (5 features: means, maxs, mins, medians, stds))
    
    """
    # Reshape the reshaped_data to split them every x sample points
    reshaped_data = np.reshape(data, (data.shape[0], -1, x))
    
    #calculate fft of reshaped_data
    reshaped_data_fft = np.abs(np.fft.fft(reshaped_data, axis=-1))

    # calculate statistical features of fft reshaped_data
    # shape:[11*(2646000/x)*x]-> shape:[11*(2646000/x)]
    fft_means = np.mean(reshaped_data_fft, axis=-1)
    fft_maxs = np.max(reshaped_data_fft, axis=-1)
    fft_mins = np.min(reshaped_data_fft, axis=-1)
    fft_medians = np.median(reshaped_data_fft, axis=-1)
    fft_stds = np.std(reshaped_data_fft, axis=-1)

    # integrade reshaped_data to a high dimensional feature matrix: size: (n_speeds * n_time_window) * (10(5 original 
    # features + 5 fft features)); 
    # sequence of features: mic_means,mic_maxs...fft_means,fft_maxs...
    fft_feats = []
    for i in [fft_means, fft_maxs, fft_mins, fft_medians, fft_stds]:
        i = i.flatten()
        fft_feats.append(i)

    fft_feats = np.array(fft_feats).T

    return fft_feats

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