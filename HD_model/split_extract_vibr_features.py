import numpy as np

def extract_fft_features(data, x):
    """ 
    extract statistical features of fft data for every x sample points

    return:
    fft feature map (type: np array; 
    size: (n_files * n_time_window(fs*secs/x)) * (5 features: means, maxs, mins, medians, stds))
    
    """
    #calculate fft of data
    reshaped_data_fft = np.abs(np.fft.fft(reshaped_data, axis=2))

    # calculate statistical features of fft data
    # shape:[11*(2646000/x)*x]-> shape:[11*(2646000/x)]
    mic_fft_means = np.mean(reshaped_data_fft, axis=-1)
    mic_fft_maxs = np.max(reshaped_data_fft, axis=-1)
    mic_fft_mins = np.min(reshaped_data_fft, axis=-1)
    mic_fft_medians = np.median(reshaped_data_fft, axis=-1)
    mic_fft_stds = np.std(reshaped_data_fft, axis=-1)

    # integrade data to a high dimensional feature matrix: size: (n_speeds * n_time_window) * (10(5 original 
    # features + 5 fft features)); 
    # sequence of features: mic_means,mic_maxs...mic_fft_means,mic_fft_maxs...
    mic_feats = []
    for i in [mic_means, mic_maxs, mic_mins, mic_medians, mic_stds, 
              mic_fft_means, mic_fft_maxs, mic_fft_mins, mic_fft_medians, mic_fft_stds]:
        i = i.flatten()
        mic_feats.append(i)

    mic_feats = np.array(mic_feats).T

    return



