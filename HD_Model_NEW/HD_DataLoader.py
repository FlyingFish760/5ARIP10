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
import scipy.io.wavfile as wav
from torch.utils.data import Dataset
from scipy.fft import fft
import torch

##########################################################
# Class to load all data into a database
##########################################################

class HD(object):
    def __init__(self, data_path=''):
        """
        Class constructor
        :param data_path: The path to data
        """
        self._year = '2023_05'
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
                'HD_status': int --> Faulty=1, Normal=0
                'Speed': int [rad/s]
                'test_iter': str
            'Microphone (str)': {
                'SampleRate': int
                'Data': list([left right]) (float)
            'Vibration (str)': {
                'SampleRate': int
                'time': list(float)
                'X_vibr': list(float)
                'Y_vibr': list(float)
                'Z_vibr': list(float)
            'dSpace (str)': {
                'SampleRate': int
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
            database[ids]['Microphone'] = self._load_mic_data(database[ids]['attributes'])
            database[ids]['Vibration'] = self._load_vibr_data(database[ids]['attributes'])
            database[ids]['dSpace'] = self._load_dSpace_data(i)

            ids=ids+1

        return database

    def generate_database_test_train(self):
    
        """
        Generates and saves the same database as the function above. But then puts N1 and F1 into training set and N2 and F2 in the testing set
            
        :return: A database dictionary
        """
        print('---------------------------------------------------------')
        print("Generating database for HD's")

        db_train = {}
        db_test = {}
        
        ids_test = 0
        ids_train = 0

        for i in listdir(self._dSpace_path):
            # Get the file name without the extension
            base_name = splitext(i)[0]
            words = base_name.split('_')

            if words[0].endswith('1'): #put it into the test set
                db_train[ids_train] = {}
                db_train[ids_train]['attributes'] = self._get_attributes(i)
                
                #Get sensor data
                db_train[ids_train]['Microphone'] = self._load_mic_data(db_train[ids_train]['attributes'])
                db_train[ids_train]['Vibration'] = self._load_vibr_data(db_train[ids_train]['attributes'])
                db_train[ids_train]['dSpace'] = self._load_dSpace_data(i)
                ids_train=ids_train+1
            
            else: #put it into the test set
                db_test[ids_test] = {}
                db_test[ids_test]['attributes'] = self._get_attributes(i)
                
                #Get sensor data
                db_test[ids_test]['Microphone'] = self._load_mic_data(db_test[ids_test]['attributes'])
                db_test[ids_test]['Vibration'] = self._load_vibr_data(db_test[ids_test]['attributes'])
                db_test[ids_test]['dSpace'] = self._load_dSpace_data(i)
                ids_test=ids_test+1

        return db_train, db_test

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
    

    def _load_mic_data(self,attributes):
        """
        Generate ...
        :return: 'Microphone (str)': {
                'SampleRate': int
                'Data': list([left right]) (float)
        """
        file_name = self._get_correct_file(self._mic_path,attributes)
        
        mic_data = {}    

        sr, data = wav.read(join(self._mic_path, file_name)) 
        
        mic_data['SampleRate'] = sr
        mic_data['Data'] = data

        return mic_data
    
    
    def _load_dSpace_data(self,i):
        """
        Generate ...
        :return:  'dSpace (str)': {
                'SampleRate': int
                'time': list(float)
                'I_bat': list(float) [A]
                'I_dyno_LP': list(float)  [A]   
                'i_motor_LP': list(float) [A]
                'w_out': list(float) [V]
                'u_bat': list(float) [V]
        """
       
        file_name = i

        arr = np.loadtxt(join(self._dSpace_path,file_name),
                 delimiter=",", dtype=str, usecols=(1,2,3,4,5,6), skiprows=16)
        data = arr[12:] # data consists of ['time','I_bat[A]', 'I_dyno_LP[A]', 'i_motor_LP[A]','speed_motor[rads]', 'u_bat[V]']
        data = data.astype(float)

        dSpace_data = {}
        dSpace_data['SampleRate'] = 500
        dSpace_data['time'] = data[:,0]           
        dSpace_data['I_bat'] = data[:,1]           
        dSpace_data['I_dyno_LP'] = data[:,2]           
        dSpace_data['i_motor_LP'] = data[:,3]           
        dSpace_data['speed_motor'] = data[:,4]           
        dSpace_data['u_bat'] = data[:,5]           

        return dSpace_data
    
    
    def _load_vibr_data(self,attributes):
        """
        Generate ...
        :return:  'Vibration (str)': {
                'SampleRate': int
                'time': list(float) # arrays not lists (which is good)
                'X_vibr': list(float)
                'Y_vibr': list(float)
                'Z_vibr': list(float)
        """
        
        file_name = self._get_correct_file(self._vibr_path,attributes)
       
        arr = np.loadtxt(join(self._vibr_path, file_name),
                 delimiter=",", dtype=str) 
        
        vibr_data = {}
        vibr_data['SampleRate'] = 3200

        data = arr[1:]
        data = data.astype(float)
        
        vibr_data['time'] = data[:,0]   
        vibr_data['X_vibr'] = data[:,1]   
        vibr_data['Y_vibr'] = data[:,2]   
        vibr_data['Z_vibr'] = data[:,3]   

        return vibr_data

    
    def _get_correct_file(self,folder,attributes):
        
        file_name = attributes

        for i in listdir(folder):
            base_name = splitext(i)[0]

            # Split the file name into individual words
            words = base_name.split('_')

            if attributes['HD_label']in words and attributes['speed'] in words and attributes['test_iter'] in words: 
                file_name = i
        
        if file_name == attributes:
            print('the following attributes cannot be found in the vibration folder: \n')
            print(attributes)

        return file_name

##########################################################
# Class to easily access splitted data
##########################################################
class SplitNormalFaulty(object):
    def __init__(self, database):
        self.normal = 0
        self.faulty = 1
        self.database = database

    def split_faulty_working_mic(self):
        """
        gives two lists, working harmonic drives and faulty harmonic drives

        input: 
            attribute: str
                'Data': list([left right]) (float)

        return : 'normal':array(float)
                'faulty':array(float)
        """
        normal = []
        faulty = []

        for data in self.database:
            if self.database[data]['attributes']['HD_status']== self.normal: #it is normal HD
                normal.append((self.database[data]['Microphone']['Data'][:,0]))
            elif self.database[data]['attributes']['HD_status']==self.faulty:
                faulty.append((self.database[data]['Microphone']['Data'][:,0]))
            else:
                print("Undefined state")

        return normal, faulty
    
    def split_faulty_working_vib(self, atrribute):
        """
        gives two lists, working harmonic drives and faulty harmonic drives

        input: 
            attribute: str
                'X_vibr': list(float)
                'Y_vibr': list(float)
                'Z_vibr': list(float)

        return : 'normal':array(float)
                'faulty':array(float)
        """
        normal = []
        faulty = []
        for data in self.database:
            if self.database[data]['attributes']['HD_status']== self.normal: #it is normal HD
                normal.append((self.database[data]['Vibration'][atrribute]))
            elif self.database[data]['attributes']['HD_status']==self.faulty:
                faulty.append((self.database[data]['Vibration'][atrribute]))
            else:
                print("Undefined state")

        return normal, faulty
    
    def split_faulty_working_dspace(self, atrribute):
        """
        gives two lists, working harmonic drives and faulty harmonic drives

        input: 
            attribute: str
                'I_bat': list(float) [A]
                'I_dyno_LP': list(float)  [A]   
                'i_motor_LP': list(float) [A]
                'w_out': list(float) [V]
                'u_bat': list(float) [V]

        return : 'normal':array(float)
                'faulty':array(float)
        """
        normal = []
        faulty = []
        for data in self.database:
            if self.database[data]['attributes']['HD_status']== self.normal: #it is normal HD
                normal.append((self.database[data]['dSpace'][atrribute]))
            elif self.database[data]['attributes']['HD_status']==self.faulty:
                faulty.append((self.database[data]['dSpace'][atrribute]))
            else:
                print("Undefined state")

        return normal, faulty
    
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.labels[idx]
        return data, label
    
class DataPreprocessing(object):
    def __init__(self, working, faulty):
        self.working = working
        self.faulty = faulty

        self.working_label = 0
        self.faulty_label = 1
        
    def create_data_batches(self, start_time = 0, stop_time = 1, window_length = 100, stride = 100):

        self.start_time = start_time
        self.stop_time = stop_time
        self.window_length = window_length
        self.stride = stride

        batches = []
        labels = []
        working_batches, working_labels = self._create_batches(self.working, label = self.working_label)

        faulty_batches, faulty_labels = self._create_batches(self.faulty, label = self.faulty_label)

        batches = working_batches + faulty_batches
        labels = working_labels + faulty_labels

        return batches, labels

    def _create_batches(self, data, label):
        data_length = len(data[0])
        self.start_sample = round(self.start_time * data_length)
        self.stop_sample = round(self.stop_time * data_length)

        batches = []
        labels = []

        for data_loop in data: #Kutnaam
            for i in range(self.start_sample,self.stop_sample,self.stride):
                batches.append(data_loop[i:i+self.window_length])
                labels.append(label)

        return batches, labels
    
##########################################################
# Class to get the data into statistical features dataset
##########################################################

class StatisticalFeaturesDataset(Dataset):
    def __init__(self, database, window_sec=2, selected_domain = {'Mic': 'time', 'dSpace': 'time'}, Mic_bool = 'True', Curr_bool = 'True'):
        self.mic_fs = database[0]['Microphone']['SampleRate']
        self.vibr_fs = database[0]['Vibration']['SampleRate']
        self.curr_fs = database[0]['dSpace']['SampleRate']
        self.N_splits = round(60/window_sec)
        self.working_label = 0
        self.faulty_label = 1
        self.selected_domain = selected_domain
        self.Mic_bool = Mic_bool
        self.Curr_bool = Curr_bool
        self.data, self.labels = self._get_statistical_features(database)
   
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label
    
    def _get_statistical_features(self, database):
        """
        Generate ...
        return:  
        """

        statistical_features = []
        labels = []

        for data in database:  
            data_mic = database[data]['Microphone']['Data'][:,0] 
            splits_mic = np.array_split(data_mic, self.N_splits)
        
            data_curr = database[data]['dSpace']['i_motor_LP']
            splits_curr = np.array_split(data_curr, self.N_splits)
            
            for i in range(self.N_splits):
                split_i_mic = splits_mic[i]
                split_i_cur = splits_curr[i]

                if not self.selected_domain['Mic']== 'time':
                    split_mic_fft = fft(split_i_mic)
                    split_i_mic = 2.0/(split_i_mic.shape[0]) * np.abs(split_mic_fft[0:split_i_mic.shape[0]//2]) 

                if not self.selected_domain['dSpace']== 'time': 
                    split_cur_fft = fft(split_i_cur)
                    split_i_cur = 2.0/(split_i_cur.shape[0]) * np.abs(split_cur_fft[0:split_i_cur.shape[0]//2]) 
 
                #Calculate the features
                mean_mic,stds_mic,maxs_mic,mins_mic,medians_mic = self._calc_statistics(split_i_mic)
                mean_cur,stds_cur,maxs_cur,mins_cur,medians_cur = self._calc_statistics(split_i_cur)
                speed = float(database[data]['attributes']['speed'])
                
                if self.Mic_bool and self.Curr_bool:
                    tot_stats = [speed, mean_mic, stds_mic, maxs_mic, mins_mic, medians_mic, mean_cur, stds_cur, maxs_cur, mins_cur, medians_cur]
                elif self.Mic_bool:
                    tot_stats = [speed, mean_mic, stds_mic, maxs_mic, mins_mic, medians_mic]
                else:
                    tot_stats = [speed, mean_cur, stds_cur, maxs_cur, mins_cur, medians_cur]

                statistical_features.append(torch.Tensor(tot_stats))
                labels.append(database[data]['attributes']['HD_status'])

        return statistical_features, labels
    
    def _calc_statistics(self,data):

        means = np.mean(data)
        stds = np.std(data)
        maxs = np.max(data)
        mins = np.min(data)
        medians = np.median(data)

        return means, stds, maxs, mins, medians

##########################################################
# Class to get the data into dataset
##########################################################

class VairableSensorsDataset(Dataset):
    def __init__(self, database, mic_bool = 'True', vibr_bool = 'False', cur_bool = 'False', start_percentage=0.1, stop_percentage = 0.9, window_sec = 1, stride_sec=0.2):
        self.mic_bool = mic_bool
        self.vibr_bool = vibr_bool
        self.cur_bool = cur_bool
        # self.speed_bool = speed_bool
        self.start_perc = start_percentage
        self.stop_perc = stop_percentage
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        
        self.mic_fs = database[0]['Microphone']['SampleRate']
        self.vibr_fs = database[0]['Vibration']['SampleRate']
        self.curr_fs = database[0]['dSpace']['SampleRate']
        self.working_label = 0
        self.faulty_label = 1

        self.data, self.labels = self._preprocess(database)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.labels[idx]
        return data, label
    
    def _get_melspectogram_features(self, data,
                    sr,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
        """
        convert file_name to a vector array.

        file_name : str
            target .wav file

        return : numpy.array( numpy.array( float ) )
            vector array
            * dataset.shape = (dataset_size, feature_vector_length)
        """
        # calculate the number of dimensions
        dims = n_mels * n_frames

        mel_spectrogram = librosa.feature.melspectrogram(y=data,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power)

        # convert melspectrogram to log mel energies
        log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

        # calculate total vector size
        n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

        # skip too short clips
        if n_vectors < 1:
            return np.empty((0, dims))

        # generate feature vectors by concatenating multiframes
        vectors = np.zeros((n_vectors, dims))
        for t in range(n_frames):
            vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

        return vectors


    def _preprocess(self,database):
        input_data = []
        labels = []
        for data in database:
            time = database[data]['dSpace']['time']
            data_mic_logmel = self._get_melspectogram_features(database[data]['Microphone']['Data'][:,0], sr=self.mic_fs, n_mels = 128, n_frames = 1, 
                                                               n_fft = round(self.window_sec*self.mic_fs), hop_length = round(self.stride_sec*self.mic_fs), power=2.0)
            data_mic_logmel_length = len(data_mic_logmel[:,0])
            counter = round((1/data_mic_logmel_length) * self.start_perc)
            for t in np.arange(round(self.start_perc*time[-1]),round(self.stop_perc*time[-1]),self.stride_sec): 
                if self.mic_bool=='True':
                    # data_mic = database[data]['Microphone']['Data'][:,0]         
                    # data_mic = data_mic[i:i+round(self.window_sec*self.mic_fs)]
                    data_mic = data_mic_logmel[counter,:]
                else:
                    data_mic = []

                if self.vibr_bool=='True':
                    i = round(t*self.vibr_fs)
                    data_vibr_X = database[data]['Vibration']['X_vibr']
                    data_vibr_X = data_vibr_X[i:i+round(self.window_sec*self.vibr_fs)]

                    data_vibr_Y = database[data]['Vibration']['Y_vibr']
                    data_vibr_Y = data_vibr_Y[i:i+round(self.window_sec*self.vibr_fs)]

                    data_vibr_Z = database[data]['Vibration']['Z_vibr']
                    data_vibr_Z = data_vibr_Z[i:i+round(self.window_sec*self.vibr_fs)]

                    data_vibr = np.concatenate((data_vibr_X, data_vibr_Y,data_vibr_Z))
                else:
                    data_vibr = []

                if self.cur_bool =='True':
                    i = round(t*self.curr_fs)
                    data_curr = database[data]['dSpace']['i_motor_LP']
                    data_curr = data_curr[i:i+round(self.window_sec*self.curr_fs)]
                else:
                    data_curr = []

                # if self.speed_bool == 'True':
                #     speed = database[data]['attributes']['speed']
                # else:
                #     speed = []

                #Combine all the sensor data that should be included
                # data_tot = np.concatenate((data_mic,data_vibr,data_curr,speed))
                data_tot = np.concatenate((data_mic,data_vibr,data_curr))
                
                #Append to the list
                input_data.append(torch.Tensor(data_tot))
                labels.append(database[data]['attributes']['HD_status'])
                counter = counter+1
            print(counter)

        return input_data, labels
