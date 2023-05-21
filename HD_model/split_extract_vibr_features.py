import numpy as np

def split_extract_mic_features(data, x):
    """ 
    Calculate the statistical features for every x sample points
    
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

    """ # use np.squeeze to eliminate the dimension with only one element
    mic_fft_means = np.squeeze(mic_fft_means)
    mic_fft_maxs = np.squeeze(mic_fft_maxs)
    mic_fft_mins = np.squeeze(mic_fft_mins)
    mic_fft_medians = np.squeeze(mic_fft_medians)
    mic_fft_stds = np.squeeze(mic_fft_stds) """
    


    # calculate statistical features of original data
    # shape:[11*(2646000/x)*x]-> shape:[11*(2646000/x)]
    mic_means = np.mean(reshaped_data, axis=-1)
    mic_maxs = np.max(reshaped_data, axis=-1)
    mic_mins = np.min(reshaped_data, axis=-1)
    mic_medians = np.median(reshaped_data, axis=-1)
    mic_stds = np.std(reshaped_data, axis=-1)

    """ # use np.squeeze to eliminate the dimension with only one element
    mic_means = np.squeeze(mic_means)
    mic_maxs = np.squeeze(mic_maxs)
    mic_mins = np.squeeze(mic_mins)
    mic_medians = np.squeeze(mic_medians)
    mic_stds = np.squeeze(mic_stds) """

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