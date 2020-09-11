# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 07:29:27 2020

@author: Taylor Royalty composed the script for preprocessing. James Senter developed the model
"""

# =============================================================================
# sep -- the separator defining fields in the sample file
# max_len -- length of sequences to be evaluated
# sample_frac -- fraction of data allocated for training and validation datasets.
#                the remaining data is allocated for test data
# embed_size -- the size of the LTSM embedding space 
# batch_size -- the number of random samples used per training iteration
# epochs -- number of times to evaluate the training/validation datasets
# data_path -- a file path to dataset to fit a model on. The file should be
#              formated to include a sequence column and an annotation column.
#              The default seperator is tab-separated
# cnn_fun_path -- directory containing cnn_functions.py 
# seq_type -- a string specifying the sequence type. This is passed to the seq_one_hot function 
#             Options are: 'aa' (iupac), 'dna', 'rna', and 'dna_iupac'
# seq_resize -- a boolean specifying whether sequences are resized or padded with 0's during
#               one-hot encoding
# mode_save_path -- file path for saving model. If empty, no model is saved.
# model_name -- a string specifying the model name when saved. 
# =============================================================================

#Input Variables
sep='\t'
max_len=1500
sample_n=3
embed_size = 256
batch_size=100
epochs=5
data_path='test_data/test_100_aa_sequences_5_classes.txt'
cnn_fun_path='scripts/python/base_model/'
seq_type='aa'
seq_resize=True 
model_save_path='data/models/'
model_name='test'

##libraries/modules
from keras.utils import to_categorical
from numpy import array

#add path to cnn_functions for importing
import sys
sys.path.insert(1,cnn_fun_path)

import cnn_functions as cf
import pandas as pd

##read in sequences as data.frame
seq_df=pd.read_csv(data_path,sep=sep)

##convert annotations into numerical categories for model response data (does not take characters)
uniq_anno=seq_df.annotation.unique() #identify unique annotations
num_classes=len(uniq_anno) #number of unique annotations
annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno}) #map numbers to unique annotations
seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation') #merge numerical categories with original dataframe

##Generate training, validation, and test datasets -- sampling occurs on annotations
train_data=seq_df.groupby(['annotation']).sample(n=sample_n) #training
seq_df=seq_df.drop(train_data.index) #remove samples in training dataset from seq_df -- do not use training data for validation/test datasets

val_data=seq_df.groupby(['annotation']).sample(n=sample_n) #validation 
seq_df=seq_df.drop(val_data.index) #remove samples in validation dataset from seq_df -- again, do not use validation data for test datasets

test_data=seq_df # take remaining sequences as the test data


del seq_df #free some memory -- seq_df no longer used

##One-hot encode x values for datasets
train_one_hot=cf.seq_one_hot(train_data['sequence'],
                             seq_type=seq_type,
                             max_len=max_len,
                             seq_resize=seq_resize) #training
val_one_hot=cf.seq_one_hot(val_data['sequence'],
                             seq_type=seq_type,
                             max_len=max_len,
                             seq_resize=seq_resize) #validation
test_one_hot=cf.seq_one_hot(test_data['sequence'],
                             seq_type=seq_type,
                             max_len=max_len,
                             seq_resize=seq_resize) #test

#One-encode y values
ytrain=to_categorical(array(train_data.ydata,dtype='uint8'),num_classes) #training
yval=to_categorical(array(val_data.ydata,dtype='uint8'),num_classes) #validation
ytest=to_categorical(array(test_data.ydata,dtype='uint8'),num_classes) #test

if seq_type == 'aa':
    #model for amino acids
    num_letters=26
    model_blstm=cf.aa_blstm(num_classes=num_classes,
                                  num_letters=num_letters,
                                  sequence_length=max_len,
                                  embed_size=embed_size)
else:
    #model for DNA--has one extra convolution layer
    num_letters=15 if seq_type == 'dna_iupac' else 4
    model_blstm=cf.dna_blstm(num_classes=num_classes,
                             num_letters=num_letters,
                             sequence_length=max_len,
                             embed_size=embed_size)

#fit model with training data and tune data with validation data
model_blstm.fit(x=train_one_hot,y=ytrain,
          batch_size=batch_size,
          validation_data=(val_one_hot,yval),
          epochs=epochs)

#evaluate model performance with test data
model_blstm.evaluate(test_one_hot,ytest)

#save model if filepath to directory is specified
if not model_save_path == '': model_blstm.save(model_save_path+model_name+'.h5')