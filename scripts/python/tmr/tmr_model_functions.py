# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
import pandas as pd
cluster_dataframe_path='data/cluster_dataframes/'

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



seq_df=load_seq_dataframe(cluster_dataframe_path)
seq_cluster=seq_df.loc[seq_df['Cluster'] > -1]
train=seq_cluster.groupby(['Cluster','annotation']).sample(2)
train_one_hot=aa_one_hot(train['sequence'])








    
