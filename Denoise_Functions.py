
from scipy.fft import fft, ifft, fftfreq
import numpy as np
from scipy.signal import lfilter, filtfilt 

def smoothen_fft(data,samplingRate):
    data_filtered = []

    N = len(data)
    for y,fs in zip(data,samplingRate):
        t = np.linspace(0.0, N*fs, N, endpoint=False)
        
        #Get the fft from the data
        yf_v1 = fft(y)
        yf = 2.0/N * np.abs(yf_v1[0:N//2])
        xf = fftfreq(N, fs)[:N//2]

        #Perform smoothing
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        # yf_filter = lfilter(b, a, yf)
        yf_filter = filtfilt(b, a, yf)
        
        data_filtered.append(ifft(yf_filter))

    return data_filtered


