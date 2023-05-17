import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import os
from scipy.io import wavfile
from scipy.fftpack import fft, ifft

#plt.style.use('seaborn-poster')

samplerate, data = wavfile.read(r"output_500Hz.wav")#(r"output_speed10_test_motion10_drive_state0.wav")
# # check length of list
# print(samplerate)
N=len(data)
print(N)
T = 1/samplerate
X = fft(data)

freq = fftfreq(N, T)[:N//2]
plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.plot(freq, 2.0/N * np.abs(X[0:N//2]))
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, samplerate/2)

plt.subplot(122)
plt.plot(freq, 2.0/N * np.abs(X[0:N//2]))
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, samplerate/2)
plt.yscale('log')

plt.savefig('fft_mic.png')
