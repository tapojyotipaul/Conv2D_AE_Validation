# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:57:29 2020

@author: tapojyoti.paul
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################
import time

########################################################################
# import additional python-library
########################################################################
import numpy
import math
import numpy as np
import keras
import random

import librosa
import librosa.core
import librosa.feature

import keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Cropping2D, Conv2DTranspose, Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.backend import int_shape

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
# from tensorflow import set_random_seed

#import librosa.core
# from import
from tqdm import tqdm
import yaml
# original lib
#import common as com
#import keras_model as keras_model

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

from sklearn.externals.joblib import load, dump
from sklearn import preprocessing

# set seed
########################################################################
# set_random_seed(1234)
########################################################################
print("Loading Packages Complete")
########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))

def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list of wav files for training
    """
    print("target_dir : {}".format(target_dir))

    # generate training list
    if dir_name==None:
        training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=target_dir, ext=ext))
    else: 
        training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        print("no_wav_file!!")

    print("train_file num : {num}".format(num=len(files)))
    return files

def file_to_vector_array(file_name, feature_range, scaler=None, 
                         sr=16000,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=1.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    y, _ = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array = log_mel_spectrogram.T

    vector_array = scaler.transform(vector_array)

    return vector_array

def list_to_vector_array(file_list, feat_path, feature_range, scaler, 
                            msg="calc...",
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    convert the file_list to features and save features.
    file_to_vector_array() is iterated, and the output vector array is saved.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    """
    ###### uncomment to compute scaler
    #scaler = preprocessing.StandardScaler()

    
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx], feature_range, scaler, 
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        ###### uncomment to compute scaler
        #scaler.partial_fit(X=vector_array)
        
        if idx == 0:
            X = np.empty((len(file_list), vector_array.shape[0], vector_array.shape[1]))
        X[idx,] = vector_array
    return X
########################################################################~
# Data Loader
########################################################################
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,128), shuffle=True, step=8):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        
        # load scaler
        scaler_file_path = "{scalers}/{machine_type}".format(scalers=param["scalers_directory"], machine_type=machine_type)
        scaler_file_path = os.path.abspath(scaler_file_path)
        #scaler = load(scaler_file_path+"/scaler_{machine_type}.bin".format(machine_type=machine_type))
        try:
            scaler = load(scaler_file_path+"/scaler_{machine_type}.bin".format(machine_type=machine_type))
        except:
            scaler = None
        self.data = list_to_vector_array(list_IDs, features_dir_train, [], scaler,
                                        msg="generate train dataset",
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"]) 
        #self.data = np.load(self.list_IDs[0] , mmap_mode='r')
        
        self.step = step
        self.indexes_start = np.arange(self.data.shape[1]-self.dim[0]+self.step, step=self.step)
        self.max = len(self.indexes_start)
        self.indexes = np.arange(self.data.shape[0])
        
        self.indexes = np.repeat(self.indexes, self.max )
        self.indexes_start = np.repeat(self.indexes_start, self.data.shape[0])
    
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] * self.max  / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes_start = self.indexes_start[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indexes, indexes_start).reshape((self.batch_size, *self.dim, 1))

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.indexes_start)


    def __data_generation(self, indexes, index_start):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, (id_file, id_start) in enumerate(zip(indexes, index_start)):

            x = self.data[id_file,]
            length, mels = x.shape

            start = id_start

            start = min(start, length - self.dim[0])
            
            # crop part of sample
            crop = x[start:start+self.dim[0], :]

            X[i,] = crop
        return X
########################################################################
mode = True

print("Prelim Function Load Complete")
###########################################################################

########################################################################
def yaml_load():
    with open("conv2d.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################
# load parameter.yaml
########################################################################
param = yaml_load()
########################################################################
########################################################################
# feature extractor
########################################################################

def feature_list_to_vector_array(file_list, feat_path, feature_range, scaler, 
                            msg="calc...",
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    convert the file_list to features and save features.
    file_to_vector_array() is iterated, and the output vector array is saved.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    """
    ###### uncomment to compute scaler
    #scaler = preprocessing.StandardScaler()

    
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx], feature_range, scaler, 
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        ###### uncomment to compute scaler
        #scaler.partial_fit(X=vector_array)
        
        if idx == 0:
            X = np.empty((len(file_list), vector_array.shape[0], vector_array.shape[1]))
        X[idx,] = vector_array

    #save features 
    print("Saving to "+feat_path+"\\data.npy")
    numpy.save(feat_path+"\\data.npy", X)
        
    ###### uncomment to compute scaler
    '''
    #save scaler
    scaler_file_path = "scalers_std_add/{machine_type}".format(machine_type=machine_type)
    # make scaler directory
    os.makedirs(scaler_file_path, exist_ok=True)
    scaler_file_path = os.path.abspath(scaler_file_path)
    dump(scaler, scaler_file_path+"/scaler_{machine_type}.bin".format(machine_type=machine_type), compress=True)
    print("dump scaler")'''
           


def feature_file_list_generator(target_dir,
                        dir_name=None,
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list of wav files for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    if dir_name==None:
    	training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=target_dir, ext=ext))
    else:
    	training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files

def select_dirs(param, mode, target=None):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        print("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        print("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))

    if target != None:    # to run model only for one machine type
        def is_one_of_in(substrs, full_str):
            for s in substrs:
                if s in full_str: return True
            return False
        list_target = [target]
        dirs = [d for d in dirs if is_one_of_in(list_target, str(d))]

    return dirs

####################################################################
    
#input_img = Input(shape=(inputDim[0], inputDim[1], 1))  # adapt this if using 'channels_first' image data format
shape0_feat = 32
shape1_feat = 128
input_img = Input(shape=(shape0_feat, shape1_feat, 1))
# encoder
x = Conv2D(32, (5, 5),strides=(1,2), padding='same')(input_img)   #32x128 -> 32x64
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (5, 5),strides=(1,2), padding='same')(x)           #32x32
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (5, 5),strides=(2,2), padding='same')(x)          #16x16
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3),strides=(2,2), padding='same')(x)          #8x8
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3),strides=(2,2), padding='same')(x)          #4x4
x = BatchNormalization()(x)
x = Activation('relu')(x)

volumeSize = int_shape(x)
# at this point the representation size is latentDim i.e. latentDim-dimensional
x = Conv2D(param["autoencoder"]["latentDim"], (4,4), strides=(1,1), padding='valid')(x)
encoded = Flatten()(x)


# decoder
x = Dense(volumeSize[1] * volumeSize[2] * volumeSize[3])(encoded) 
x = Reshape((volumeSize[1], volumeSize[2], 512))(x)                #4x4

x = Conv2DTranspose(256, (3, 3),strides=(2,2), padding='same')(x)  #8x8
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(128, (3, 3),strides=(2,2), padding='same')(x)  #16x16   
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(64, (5, 5),strides=(2,2), padding='same')(x)   #32x32
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2DTranspose(32, (5, 5),strides=(1,2), padding='same')(x)   #32x64
x = BatchNormalization()(x)
x = Activation('relu')(x)

decoded = Conv2DTranspose(1, (5, 5),strides=(1,2), padding='same')(x) 

model=  Model(inputs=input_img, outputs=decoded)

model.load_weights('model_fan.hdf5') ##Change Location

model.compile(**param["fit"]["compile"])
print("Model Load Complete")

##########################################################################

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    print("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################
    
import itertools
import re
from sklearn.metrics import precision_recall_fscore_support,average_precision_score,recall_score
from sklearn import metrics
#########################################################################

precision_ids = []

#Change Location
target='fan'
dirs = ['/home/ubuntu/DenseAE/Data_root/fan']
print("dirs: ",dirs)
target_dir = "/home/ubuntu/DenseAE/Data_root/fan"
scaler_file_path = "scaler_fan.bin"
id_str = "id_06"
csv_lines = []
#######################################################3

print("\n===========================")
# print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
machine_type = os.path.split(target_dir)[1]

print("============== MODEL LOAD ==============")
# set model path
model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                        machine_type=machine_type)

features_file_path = "{features}/{machine_type}/{tip}".format(features=param["features_directory"],
                                                                machine_type=machine_type, tip="test")
features_dir_path = os.path.abspath(features_file_path)

#load scaler
try:
    scaler = load(scaler_file_path)
except:
    scaler = None


if mode:
    # results by type
    csv_lines.append([machine_type])
    csv_lines.append(["id", "AUC", "pAUC"])
    performance = []

machine_id_list = get_machine_id_list_for_test(target_dir)

test_files, y_true = test_file_list_generator(target_dir, "id_06")

# setup anomaly score file path
anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                         result=param["result_directory"],
                                                                         machine_type=machine_type,
                                                                         id_str=id_str)
anomaly_score_list = []
###########################################################################
data_all = np.empty((0, 32, 128, 1), int)
print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
y_pred = [0. for k in test_files]
for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):

    try:
        # get audio features
        vector_array = file_to_vector_array(file_path, [], scaler,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
#         print("yes")

        length, _ = vector_array.shape

        dim = param["autoencoder"]["shape0"]
        step = param["step"]
#         print("yes1")
        idex = numpy.arange(length-dim+step, step=step)

        for idx in range(len(idex)):
            start = min(idex[idx], length - dim)

            vector = vector_array[start:start+dim,:]

            vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
            if idx==0:
                batch = vector
            else:
                batch = numpy.concatenate((batch, vector))

#         print("yes1")
        # add channels dimension
        data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))
        data_all = np.append(data_all, data, axis=0)
        # calculate predictions
        # errors = numpy.mean(numpy.square(data - model.predict(data)), axis=-1)

        # y_pred[file_idx] = numpy.mean(errors)
        # anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])


    except:
        print("file broken!!: {}".format(file_path))
        sys.exit(-1)

# save anomaly score
#save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
print("anomaly score result ->  {}".format(anomaly_score_csv))

if mode:
    # append AUC and pAUC to lists
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
    csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
    performance.append([auc, p_auc])
    print("AUC : {}".format(auc))
    print("pAUC : {}".format(p_auc))
    precision = average_precision_score(y_true, y_pred)
    precision_ids.append(precision)
    print("Precision : {}".format(precision))

print("\n============ END OF TEST FOR A MACHINE ID ============")
##########################################################################
print("Inferencing Started...............")

X = data_all

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'
import pandas as pd

def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]


def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    basic_key = ["median","mean","std_dev","min_time","max_time","quantile_10","quantile_90"]
    basic_value = [median,mean,std_dev,min_time,max_time,quantile_10,quantile_90]

    dict_basic = dict(zip(basic_key, basic_value))
    
    return pd.DataFrame(dict_basic, index = [0])

import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer


import os
import sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore

plugin_dir = None
model_xml = r'.\batch_100\my_model.xml'
model_bin = r'.\batch_100\my_model.bin'

plugin_dir = None
ie = IECore()
# versions = ie.get_versions("CPU")
# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
# check net.inputs.keys(), net.outputs
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# exec_net = plugin.load(network=net)
exec_net_100 = ie.load_network(network=net, device_name="CPU")
del net

NUM_LOOPS = 10
def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = get_test_data(num_observations)
    data = test_df

    num_rows = len(test_df)
    # print(f"Running {NUM_LOOPS} inference loops with batch size {num_rows}...")

    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):

        start_time = timer()
        model.predict(data)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    print(num_observations, ", ", calculate_stats(inference_times))
    return calculate_stats(inference_times)

def run_inference_ov(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = get_test_data(num_observations)
    data = test_df.reshape(test_df.shape[0],1,32,128)

    num_rows = len(test_df)
    # print(f"Running {NUM_LOOPS} inference loops with batch size {num_rows}...")

    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):

        start_time = timer()
        i = 0
        count = 0
        res_all = np.empty((0, 1, 32, 128), int)
        while i<num_observations:
            res = exec_net_100.infer(inputs={input_blob: data[i:i+1000]})
            res_all = np.append(res_all, res['conv2d_transpose_5/BiasAdd/Add'], axis=0)
            i = i+1000
            count = count+1        
        end_time = timer()
        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)
    
    print("count of batches:",count)
    print(num_observations, ", ", calculate_stats(inference_times))
    return calculate_stats(inference_times)

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'

if __name__=='__main__':
    ob_ct = 1000  # Start with a single observation
    logging.info(STATS)
    temp_df = pd.DataFrame()
    while ob_ct <= 10000:
        temp = run_inference(ob_ct)
        temp["No_of_Observation"] = ob_ct
        temp_df = temp_df.append(temp)
        ob_ct *= 10
    print("Summary........")
    print(temp_df)
    print ("Below are the results for OpenVino.........")
    ob_ct = 1000  # Start with a single observation
    logging.info(STATS)
    temp_df = pd.DataFrame()
    while ob_ct <= 10000:
        temp = run_inference_cv(ob_ct)
        temp["No_of_Observation"] = ob_ct
        temp_df = temp_df.append(temp)
        ob_ct *= 10
    print("Summary OpenVino........")
    print(temp_df)
