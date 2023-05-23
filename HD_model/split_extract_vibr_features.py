def extract_fft_features(reshaped_data, x):
    """ 
    extract statistical features of fft reshaped_data for every x sample points

    return:
    fft feature map (type: np array; 
    size: (n_files * n_time_window(fs*secs/x)) * (5 features: means, maxs, mins, medians, stds))
    
    """
    # Reshape the reshaped_data to split them every x sample points
    reshaped_reshaped_data = np.reshape(reshaped_data, (reshaped_data.shape[0], -1, x))
    
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
