"""
This is the dataloader intended for the HD data with microphone, vibrational and current sensors

"""
import pickle
import sys
import librosa
import numpy as np

from os.path import join, abspath, isfile, isdir, splitext
from os import makedirs, listdir
from sklearn.model_selection import train_test_split, KFold


class HD(object):
    def __init__(self, data_path=''):
        """
        Class constructor
        :param data_path: The path to data
        """
        self._year = '2023'
        self._name = 'HD'
        self._dSpace_ext = '.csv'
        self._vibr_ext = '.csv'
        self._mic_ext = '.wav'

        # Paths
        self._pie_path = data_path if data_path else self._get_default_path()
        assert isdir(self._pie_path), \
            'pie path does not exist: {}'.format(self._pie_path)

        self._mic_path = join(self._pie_path, 'Microphone')
        self._vibr_path = join(self._pie_path, 'Vibration')
        self._dSpace_path = join(self._pie_path, 'dSpace')

   
    def generate_database(self):

        """
        Generates and saves a database of the pie dataset by integrating all annotations
        Dictionary structure:
        'test_id'(int): {
            'attributes (str)': {
                'HD_label': str --> F1,N1,...
                'HD_state': int --> Faulty=1, Normal=0
                'Speed': int [rad/s]
                'test_iter': str
            'Microphone (str)': {
                'SampleRate': int
                'data': list([left right]) (float)
            'Vibration (str)': {
                'SampleRate': int
                'time': list(float)
                'X_vibr': list(float)
                'Y_vibr': list(float)
                'Z_vibr': list(float)
            'dSpace (str)': {
                'time': list(float)
                'I_bat': list(float) [A]
                'I_dyno_LP': list(float)  [A]   
                'i_motor_LP': list(float) [A]
                'w_out': list(float) [V]
                'u_bat': list(float) [V]
            
        :return: A database dictionary
        """
        print('---------------------------------------------------------')
        print("Generating database for HD's")

        database = {}
        
        ids = 0
        for i in listdir(self._dSpace_path):

            database[ids] = {}
            database[ids]['attributes'] = self._get_attributes(i)
            
            #Get sensor data
            database[ids]['Microphone'] = self._get_mic_data(database[ids]['attributes'])
            database[ids]['Vibration'] = self._get_vibr_data(database[ids]['attributes'])
            database[ids]['dSpace'] = self._get_dSpace_data(i)

            ids=ids+1

        return database
    

    # Data loading functions 
    def _get_attributes(self,file_name):
        # Get the file name without the extension
        base_name = splitext(file_name)[0]

        # Split the file name into individual words
        words = base_name.split('_')
        
        attributes = {}
        
        #Extract correct features from the file name
        for word in words:
            if word.startswith('F'):
                attributes['HD_label'] = word
                attributes['HD_status'] = 1
            elif word.startswith('N'):
                attributes['HD_label'] = word
                attributes['HD_status'] = 0  

        attributes['speed'] = words[words.index('speed')+1]
        attributes['test_iter'] = words[words.index('test')+1]

        return attributes
    

    def _get_mic_data(self,attributes):
        """
        Generate ...
        :return:
        """
        file_name = self._get_correct_file(self._mic_path,attributes)

        mic_data = {}

        # TODO: load in the correct data

        return mic_data
    
    
    def _get_dSpace_data(self,i):
        file_name = i

        dSpace_data = {}

        # TODO: load in the correct data

        return dSpace_data
    
    
    def _get_vibr_data(self,attributes):
        file_name = self._get_correct_file(self._vibr_path,attributes)
        
        vibr_data = {}

        # TODO: load in the correct data

        return vibr_data

    
    def _get_correct_file(self,folder,attributes):
        
        for i in listdir(folder):
            base_name = splitext(i)[0]

            # Split the file name into individual words
            words = base_name.split('_')

            if attributes['HD_label']in words and attributes['speed'] in words and attributes['test_iter'] in words: 
                file_name = i

        return file_name


# def file_to_vectors(data,
#                     n_mels=64,
#                     n_frames=5,
#                     n_fft=1024,
#                     hop_length=512,
#                     power=2.0):
#     """
#     convert file_name to a vector array.

#     file_name : str
#         target .wav file

#     return : numpy.array( numpy.array( float ) )
#         vector array
#         * dataset.shape = (dataset_size, feature_vector_length)
#     """
#     # calculate the number of dimensions
#     dims = n_mels * n_frames

#     mel_spectrogram = librosa.feature.melspectrogram(y=data,
#                                                      sr=sr,
#                                                      n_fft=n_fft,
#                                                      hop_length=hop_length,
#                                                      n_mels=n_mels,
#                                                      power=power)

#     # convert melspectrogram to log mel energies
#     log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

#     # calculate total vector size
#     n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

#     # skip too short clips
#     if n_vectors < 1:
#         return np.empty((0, dims))

#     # generate feature vectors by concatenating multiframes
#     vectors = np.zeros((n_vectors, dims))
#     for t in range(n_frames):
#         vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

#     return vectors