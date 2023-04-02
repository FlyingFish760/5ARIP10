
from scipy.fft import fft, ifft, fftfreq
import numpy as np

def smoothen_fft(data,samplingRate):
    file_fft = []
    file_fft_half = []

    N = len(data)
    for y,fs in zip(data,samplingRate):
        t = np.linspace(0.0, N*fs, N, endpoint=False)
        
        #Get the fft from the data
        yf_v1 = fft(y)
        yf = 2.0/N * np.abs(yf_v1[0:N//2])
        xf = fftfreq(N, fs)[:N//2]

        #Perform smoothing
    
    
    return data


