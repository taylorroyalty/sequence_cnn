# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:49:44 2020

@author: Peng
"""

#%%
import os
import pandas as pd

from random import shuffle
from keras.models import Model
from sklearn.manifold import TSNE
from numpy import zeros
from keras.models import Sequential
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
from keras.optimizers import Adam#, Nadam

def seq_one_hot(seqs,seq_type='aa',max_len=None,seq_resize=True):
# =============================================================================
# one-hot encodes sequences for use in nn modeling.
# seqs -- a list where each element is a biological sequence as a string 
# seq_type -- specifies type of biological sequence. Support options are ['aa', 'dna', 'rna', 'dna_iupac']. 
#             defualt: 'aa'
# max_len -- specifies the length of sequences. Defualt is None. This takes the maximum length sequence as the max.
# seq_resize -- This option resizes sequences using tensorflow.image resize  
# =============================================================================

    #create dictionary matching sequences positions to feature index for one-hot encoded matrix
    if seq_type == 'aa':
        seq_dict= {"A": 0,
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
        n_letter=26
    elif seq_type == 'dna':
        seq_dict= {"A": 0,
                  "T": 1,
                  "C": 2,
                  "G": 3}
        n_letter=4
    elif seq_type == 'dna_iupac':
        seq_dict= {"A": 0,
                  "T": 1,
                  "C": 2,
                  "G": 3,
                  "Y": 4,
                  "S": 5,
                  "W": 6,
                  "K": 7,
                  "M": 8,
                  "B": 9,
                  "D": 10,
                  "H": 11,
                  "V": 12,
                  "N": 13,
                  "-": 14}
        n_letter=15
        
    else:
        return "Supported seq_type options include: ['aa', 'dna', 'rna', 'dna_iupac']"
    
    #find maximum length sequence
    if max_len == None:
        n=[len(seq) for seq in seqs]
        max_len = max(n)
    
    #pre-define numpy matrix based on length and number of sequences
    one_hot_matrix=zeros(shape=(len(seqs),max_len,n_letter),dtype='float')    
    
    #indexing matching bases/aa's to dictionary and populating one_hot_matrix
    #feature index is retrieved with dictionary
    if seq_resize == True:
        from tensorflow.image import resize
        
        i=0
        for seq in seqs: #loop through each sequence in list seqs
            j=0
            tmp_vector=zeros(shape=(1,len(seq),n_letter,1)) #define 4-D tensor with 1 sample and 1 channel
            for letter in seq: #loop through each base/aa in sequence
                indx=seq_dict[letter] #match letter to dictionary
                tmp_vector[0,j,indx,0]=1 
                j+=1
            one_hot_matrix[i,:,:]=resize(tmp_vector,size=(max_len,n_letter))[0,:,:,0].numpy() #reshape 4-D tensor to 2D
            i+=1
    else:
        i=0
        for seq in seqs:
            j=0
            for letter in seq:
                indx=seq_dict[letter] #match letter to dictionary
                one_hot_matrix[i,j,indx]=1
                j+=1
                if j == max_len: break
            i+=1
            
    return one_hot_matrix

#%%
#load data from directory        
def load_seq_dataframe(dir_path):
    
    seq_df=pd.DataFrame()
    for filename in os.listdir(dir_path):
        new_csv=dir_path+filename
        seq_df=seq_df.append(pd.read_csv(new_csv))
        
    return seq_df
#%%
#model architecture for amino acids
def original_blstm(num_classes, num_letters, sequence_length, embed_size=50):
    
    model = Sequential()
    model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=100, kernel_size=26, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_size=13, strides=13))
    model.add(Masking(mask_value=0))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(LSTM(embed_size, activation="tanh"))
    model.add(Dense(num_classes, activation=None, name="AV"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
#%%
def dna_blstm(num_classes, num_letters, sequence_length, embed_size=256):
    
    model = Sequential()
    model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=26, kernel_size=3, strides=3, padding="valid", activation="relu"))
    model.add(Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_length=13, stride=13))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(LSTM(embed_size, activation="tanh"))
    model.add(Dense(num_classes, activation=None, name="AV"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

#%%
def aa_blstm(num_classes, num_letters, sequence_length, embed_size=5000):
    
    model = Sequential()
    # model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=100, kernel_size=26, padding="valid", activation="relu"))
    # model.add(MaxPooling1D(pool_size=13, strides=13))
    # model.add(Masking(mask_value=0))
    # model.add(Dropout(0.2))
    # model.add(Embedding(num_letters,10000))
    # model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(5000, dropout=0.2, recurrent_dropout=0.2, activation="tanh", return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(embed_size, activation="tanh"))
    model.add(Dense(num_classes, activation=None, name="AV"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

#%%


def tsne_non_trained_classes(model,data,write_path,layer,max_len,seq_type='aa',seq_resize=True):
        
        embed_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
        embed_model.summary()
        
        new_seq=seq_one_hot(data['sequence'],seq_type=seq_type,max_len=max_len,seq_resize=seq_resize)
        embed = embed_model.predict(new_seq, batch_size=100, verbose=1)
        tsne = TSNE(n_components=2, random_state=0)
        xx = tsne.fit_transform(embed)
        
        data['comp1']=xx[:,0]
        data['comp2']=xx[:,1]
        
        data.to_csv(write_path,sep='\t')

def randomize_groups(df,x,f=1):
# =============================================================================
# shuffles dependent variables (columns) with respect to a dataframe
# df -- a dataframe
# x -- list containing columns which are not shuffled--i.e., independent columns (string)
# f -- fraction of sample dependent columns to shuffle (float from 0 to 1) 
# =============================================================================
    
    if f>1:
        print("f ranges 0 to 1--f was set to 1")
        f=1
    elif f<0:
        print("f ranges 0 to 1--f was set to 0")
        f=0
    
    index_keep=df.sample(frac=f).index
    df_tmp=df.drop(index_keep).reset_index(drop=True)
    df=df.loc[index_keep]
    
    for col in df_tmp.columns:
        if col in x: continue 
        df_tmp[col]=df_tmp[col].sample(frac=1).reset_index(drop=True)
        
    return(df.append(df_tmp))
        
    