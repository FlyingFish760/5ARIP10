"""
This is the dataloader intended for the HD data with microphone, vibrational and current sensors

"""
import pickle
import cv2
import sys

import xml.etree.ElementTree as ET
import numpy as np

from os.path import join, abspath, isfile, isdir
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


    # Path generators
    def get___(self):
        """
        Generates a path to save cache files
        :return:
        """

        return