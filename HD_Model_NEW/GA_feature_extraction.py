
#Import libaries
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile 
import os
import csv


def extract_time_features(data, x):
    """ 
    extract statistical features of data in time domain for every x sample points

    return:
    feature map (type: np array; 
                size: (n_test_files * n_time_window: fs*secs/x) * (15 features))
    
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
                    size: (n_test_files * n_time_window:fs*secs/x) * (5 features: means, maxs, mins, medians, stds))
    
    """
    # Reshape the reshaped_data to split them every x sample points
    reshaped_data = np.reshape(data, (data.shape[0], -1, x))
    
    #calculate fft of reshaped_data
    reshaped_data_fft = np.abs(np.fft.fft(reshaped_data, axis=-1))

    # calculate statistical features of fft reshaped_data
    # shape:[n_test_files*(2646000/x)*x]-> shape:[n_test_files*(2646000/x)]
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

