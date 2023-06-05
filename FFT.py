# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from pylab import *
import os
from HD_Model_NEW.HD_DataLoader import *

dir_data = os.path.join(os.getcwd(), 'HD_Model_NEW', 'HD_data')
HD_object = HD(data_path=dir_data)
data = HD_object.generate_database()


class SensorDataProcessor:
    def __init__(self, data):
        self.data = data

    def make_fft(self, plot, fs):
        window = np.hanning(len(plot))
        plot = plot * window
        X = fft(plot)
        X = np.abs(X)
        X_db = 20 * np.log10(X)
        N = len(X)
        nfft = np.arange(N)
        T = N / fs
        freq = nfft / T

        return freq, X_db

    def make_fft_moving_mean(self, plot, fs):
        window = np.hanning(len(plot))
        plot = plot * window
        X = fft(plot)
        X = np.abs(X)
        X_db = 20 * np.log10(X)

        # making the frequency vector
        N = len(X)
        nfft = np.arange(N)
        T = N / fs
        freq = nfft / T
        X_db = np.convolve(X_db, np.ones(100) / 100, mode='same')

        return freq, X_db

    def get_concat_sensor(self, sensor, selection):
        normal = []
        faulty = []
        concat_normal = []
        concat_faulty = []
        if sensor == 'Microphone':
            for testnr in self.data:
                if self.data[testnr]['attributes']['HD_status'] == 0:  # it is normal HD
                    normal.append((self.data[testnr][sensor][selection][:, 0]))
                elif self.data[testnr]['attributes']['HD_status'] == 1:
                    faulty.append((self.data[testnr][sensor][selection][:, 0]))
                else:
                    print("Undefined state")

            normal = np.array(normal)
            faulty = np.array(faulty)

            for i in range(len(normal)):
                concat_normal = np.concatenate((concat_normal, normal[i]))
            for i in range(len(faulty)):
                concat_faulty = np.concatenate((concat_faulty, faulty[i]))

        if sensor == 'Vibration':
            for testnr in self.data:
                if self.data[testnr]['attributes']['HD_status'] == 0:
                    normal.append((self.data[testnr][sensor][selection]))
                elif self.data[testnr]['attributes']['HD_status'] == 1:
                    faulty.append((self.data[testnr][sensor][selection]))
                else:
                    print("Undefined state")

            for i in range(len(normal)):
                concat_normal = np.concatenate((concat_normal, normal[i]))
            for i in range(len(faulty)):
                concat_faulty = np.concatenate((concat_faulty, faulty[i]))

        if sensor == 'dSpace':
            for testnr in self.data:
                if self.data[testnr]['attributes']['HD_status'] == 0:  # it is normal HD
                    normal.append((self.data[testnr][sensor][selection]))
                elif self.data[testnr]['attributes']['HD_status'] == 1:
                    faulty.append((self.data[testnr][sensor][selection]))
                else:
                    print("Undefined state")

            for i in range(len(normal)):
                concat_normal = np.concatenate((concat_normal, normal[i]))

            for i in range(len(faulty)):
                concat_faulty = np.concatenate((concat_faulty, faulty[i]))

        return concat_normal, concat_faulty