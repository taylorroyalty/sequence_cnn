# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
import pandas as pd
import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation, Flatten, Input, Multiply
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam
# import sys

# sys.path.insert(1,'scripts/python/tmr/')
cluster_dataframe_path='data/cluster_dataframes/'

# from model_templates_tmr import original_blstm
from keras.utils import to_categorical
is_dna_data=False

def aa_one_hot(seqs):
# =============================================================================
# one-hot encodes amino acid sequences. Sequences shorter than the longest sequence are 
# paded with 0's for all amino acid feature. Functionality in the future will include a nucleotide
# option as well as an abilit to truncate sequences to a specified length.
# -seqs-- a list where each element is an amino acid string
# =============================================================================
    import numpy as np
    #dictionary matching AA and feature index for one-hot encoded matrix
    aa_dict= {"A": 0,
              "C": 1,
              "D": 2,
              "E": 3,
              "F": 4,
              "G": 5,
              "H": 6,
              "I": 7,
              "K": 8,
              "L": 9,
              "M": 10,
              "N": 11,
              "P": 12,
              "Q": 13,
              "R": 14,
              "S": 15,
              "T": 16,
              "V": 17,
              "W": 18,
              "Y": 19,
              "X": 20,
              "B": 21,
              "Z": 22,
              "J": 23,
              "U": 24,
              "O": 25}
    
    #find maximum length sequence
    n=[len(seq) for seq in seqs]
    #pre-define matrix based on length and number of sequences; pad 0s on end
    #of sequences shorter than maximum length sequence
    one_hot_matrix=np.zeros(shape=(len(seqs),max(n),26),dtype='uint8')    
    
    #indexing one_hot samples and timeseries (i.e., aa position)
    #feature index is retrieved with dictionary
    i=0
    for seq in seqs:
        j=0
        for aa in seq:
            a=aa_dict[aa]
            one_hot_matrix[i,j,a]=1
            j+=1    
        i+=1
    return one_hot_matrix
        
def load_seq_dataframe(dir_path):
    import os
    import pandas as pd
    
    seq_df=pd.DataFrame()
    for filename in os.listdir(dir_path):
        new_csv=dir_path+filename
        seq_df=seq_df.append(pd.read_csv(new_csv))
        
    return seq_df

def original_blstm(num_classes, num_letters, sequence_length, embed_size=50):
    model = Sequential()
    model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=320, kernel_size=26, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_size=13, strides=13))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
    model.add(Dropout(0.5))
    #model.add(LSTM(num_classes, activation="softmax", name="AV"))
    model.add(LSTM(embed_size, activation="tanh"))
    model.add(Dense(num_classes, activation=None, name="AV"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

seq_df=load_seq_dataframe(cluster_dataframe_path)

uniq_anno=seq_df.annotation.unique()
num_classes=len(uniq_anno)
annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation')
seq_cluster=seq_df.loc[seq_df['Cluster'] > -1]
train=seq_cluster.groupby(['annotation']).sample(1)



train_one_hot=aa_one_hot(train['sequence'])



num_letters = 4 if is_dna_data else 26
sequence_length = train_one_hot.shape[1]
mask_length = None

embed_size = 256
# model_name = 'testing'
# model_template = original_blstm
model = original_blstm(num_classes, num_letters, sequence_length, embed_size=embed_size)

ydata=to_categorical(np.array(train.ydata,dtype='uint8'),num_classes)
model.train_on_batch(x=train_one_hot,y=ydata)




    
