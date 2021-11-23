# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:56:04 2020

@author: Peng
"""

from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Model, load_model
import sys
import pandas as pd
import numpy as np

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

emb_data_path='data/swiss_data_variants/swiss_n1.tsv'
write_path='data/tara/swiss_n1_11182020_embedding.tsv'
model_path='/home/troyalty/Documents/projects/sequence_cnn/data/models/iteration/iteration_swiss_n1/swiss_iteration_0.h5'
sep='\t'
max_len=300
embed_size = 256
batch_size=100
seq_type='aa'
seq_resize=False
layer="lstm_1"
n_components=2

model=load_model(model_path)
emb_data=pd.read_csv(emb_data_path,sep=sep)
embed_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
#	embed_model.summary()
new_seq=cf.seq_one_hot(emb_data['sequence'],seq_type=seq_type,max_len=max_len,seq_resize=seq_resize)
embed = embed_model.predict(new_seq)
emb_data=pd.concat([emb_data,pd.DataFrame(embed)],axis=1)

emb_data.to_csv(write_path,sep='\t')
