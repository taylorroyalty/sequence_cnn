# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 07:29:27 2020

@author: Taylor Royalty composed the script for preprocessing. James Senter developed the model
"""

# =============================================================================
# max_len -- length of sequences to be evaluated
# sample_frac -- fraction of data allocated for training and validation datasets.
#                the remaining data is allocated for test data
# embed_size -- the size of the LTSM embedding space 
# batch_size -- the number of random samples used per training iteration
# epochs -- number of times to evaluate the training/validation datasets
# data_path -- a file path to dataset to fit a model on. The file should be
#              formated to include a sequence column and an annotation column.
#              The default seperator is tab-separated
# =============================================================================

#Input Variables
max_len=100
sample_frac=0.2
embed_size = 256
batch_size=100
epochs=50
data_path='data/swiss_n100.tsv'
cnn_fun_path='scripts/python/tmr/'
seq_type='aa' #options include 'aa' (iupac), 'dna', 'rna', and 'dna_iupac'

#libraries/modules
from keras.utils import to_categorical
from numpy import array

#add path to cnn_functions for importing
import sys
sys.path.insert(1,cnn_fun_path)
import cnn_functions as cf

import pandas as pd

#read in sequences as data.frame
seq_df=pd.read_csv(data_path,sep='\t')

#convert annotations into numerical categories for model response data (does not take characters)
uniq_anno=seq_df.annotation.unique() #identify unique annotations
num_classes=len(uniq_anno) #number of unique annotations
annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno}) #map numbers to unique annotations
seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation') #merge numerical categories with original dataframe

#Generate training, validation, and test datasets -- sampling occurs on annotations
train_data=seq_df.groupby(['annotation']).sample(frac=sample_frac) #training
seq_df=seq_df.drop(train_data.index) #remove samples in training dataset from seq_df -- do not use training data for validation/test datasets

val_data=seq_df.groupby(['annotation']).sample(frac=sample_frac) #validation 
seq_df=seq_df.drop(train_data.index) #remove samples in validation dataset from seq_df -- again, do not use validation data for test datasets

test_data=seq_df # take remaining sequences as the test data

#free some memory -- seq_df no longer used
del seq_df 

#One-hot encode x values for datasets
train_one_hot=cf.seq_one_hot(train_data['sequence'],max_len=max_len) #training
val_one_hot=cf.seq_one_hot(val_data['sequence'],max_len=max_len) #validation
test_one_hot=cf.seq_one_hot(test_data['sequence'],max_len=max_len) #test

#One-encode y values
ytrain_a=to_categorical(array(train_data.ydata,dtype='uint8'),num_classes) #training
yvalidation_a=to_categorical(array(val_data.ydata,dtype='uint8'),num_classes) #validation
ytest_a=to_categorical(array(test_data.ydata,dtype='uint8'),num_classes) #test


