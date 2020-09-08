# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:49:44 2020

@author: Peng
"""

#%%

def aa_one_hot(seqs,max_len=None):
# =============================================================================
# one-hot encodes amino acid sequences. Sequences shorter than the longest sequence are 
# paded with 0's for all amino acid features. Functionality in the future will include a nucleotide
# option.
# -seqs-- a list where each element is an amino acid string
# =============================================================================
    import numpy as np
    from tensorflow.image import resize
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
    if max_len == None:
        n=[len(seq) for seq in seqs]
        max_len = max(n)
    
    #pre-define matrix based on length and number of sequences; pad 0s on end
    #of sequences shorter than maximum length sequence
    one_hot_matrix=np.zeros(shape=(len(seqs),max_len,26),dtype='float')    
    
    #indexing one_hot samples and timeseries (i.e., aa position)
    #feature index is retrieved with dictionary
    i=0
    for seq in seqs:
        j=0
        tmp_vector=np.zeros(shape=(1,len(seq),26,1))
        for aa in seq:
            a=aa_dict[aa]
            tmp_vector[0,j,a,0]=1
            # one_hot_matrix[i,j,a]=1
            j+=1
            # if j == max_len: break
        one_hot_matrix[i,:,:]=resize(tmp_vector,size=(max_len,26))[0,:,:,0].numpy()
        i+=1
    return one_hot_matrix

#%%
#load data from directory        
def load_seq_dataframe(dir_path):
    import os
    import pandas as pd
    
    seq_df=pd.DataFrame()
    for filename in os.listdir(dir_path):
        new_csv=dir_path+filename
        seq_df=seq_df.append(pd.read_csv(new_csv))
        
    return seq_df
#%%
#model architecture
def original_blstm(num_classes, num_letters, sequence_length, embed_size=50):
    from keras.models import Sequential
    from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation
    # from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam#, Nadam
    
    model = Sequential()
    model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=320, kernel_size=26, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_size=13, strides=13))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=100, kernel_size=26, padding="valid", activation="relu"))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Masking(mask_value=0))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
    model.add(Dropout(0.5))
    #model.add(LSTM(num_classes, activation="softmax", name="AV"))
    model.add(LSTM(embed_size, activation="tanh"))
    model.add(Dense(num_classes, activation=None, name="AV"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

#%%

def tsne_non_trained_classes(model,data,write_path,layer,max_len):
        import pandas as pd
        from keras.models import Model
        from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        
        embed_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
        embed_model.summary()
        
        new_seq=aa_one_hot(data['sequence'],max_len=max_len)
        embed = embed_model.predict(new_seq, batch_size=100, verbose=1)
        tsne = TSNE(n_components=2, random_state=0)
        xx = tsne.fit_transform(embed)
        
        data['comp1']=xx[0,:]
        data['comp2']=xx[1,:]
        
        data.to_csv(write_path)
