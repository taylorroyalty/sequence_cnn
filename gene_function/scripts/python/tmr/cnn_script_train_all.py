# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
#%%
# Libraries and modules
import pandas as pd
import numpy as np
from keras.utils import to_categorical

import sys

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf
#%%
#Inputs
layer="lstm_1"
tnse_write_path='data/tnse_results/test.tsv'
emb_data_path='data/swiss_1_99.tsv'
data_path='data/cluster_dataframes/'
model_save_path='data/models/'
new_model=True

#nn parameters
max_len=1500
sample_n=10
embed_size = 256
batch_size=100
epochs=100
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
seq_resize=True 

#%%
#generate datasets for fitting
if new_model == True:
    seq_df=cf.load_seq_dataframe(data_path)
    uniq_anno=seq_df.annotation.unique()
    num_classes=len(uniq_anno)
    annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
    seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation').drop(['Component_1','Component_2','Cluster','id','annotation'],axis=1)
    
    seq_df=randomize_groups(seq_df,'ydata',f)
    
    #%%
    #generate training data for annotation/cluster datasets
    ##annotations
    train_a=seq_df.groupby(['annotation']).sample(n=sample_n)
    seq_df=seq_df.drop(train_a.index)
    train_a_one_hot=cf.seq_one_hot(train_a['sequence'],
                                  seq_type=seq_type,
                                  max_len=max_len,
                                  seq_resize=seq_resize)
    
    #%%
    #generate validation data for annotation/cluster datasets
    ##annotation
    validation_a=seq_df.groupby(['annotation']).sample(n=sample_n)
    seq_df=seq_df.drop(validation_a.index)
    validation_a_one_hot=cf.seq_one_hot(validation_a['sequence'],
                                  seq_type=seq_type,
                                  max_len=max_len,
                                  seq_resize=seq_resize)
    
    #generate test data for annotation/cluster datasets
    ##annotation
    
    test_a=seq_df
    test_a_one_hot=cf.seq_one_hot(test_a['sequence'],
                                  seq_type=seq_type,
                                  max_len=max_len,
                                  seq_resize=seq_resize)
    
    #%%
    ##generate y data for annotation/cluster datasets
    ##annotation
    ytrain_a=to_categorical(np.array(train_a.ydata,dtype='uint8'),num_classes)
    yvalidation_a=to_categorical(np.array(validation_a.ydata,dtype='uint8'),num_classes)
    ytest_a=to_categorical(np.array(test_a.ydata,dtype='uint8'),num_classes)
    
    # sequence_length = train_one_hot.shape[1]
    model_a= cf.original_blstm(num_classes,
                               num_letters,
                               max_len,
                               embed_size=embed_size)
    
    n_validation_a=validation_a_one_hot.shape[0]
    
    model_a.fit(x=train_a_one_hot,y=ytrain_a,batch_size=batch_size,
                validation_data=(validation_a_one_hot,yvalidation_a),
                epochs=epochs)
    
    
    model_a.save(model_save_path + 'swiss100_annotation_only.h5')
    
    model_a.evaluate(test_a_one_hot,ytest_a)
    
else:
    from keras.models import load_model
    model_c=load_model('data/models/swiss100_clusters.h5')

emb_data=pd.read_csv(emb_data_path,sep='\t').groupby("annotation").filter(lambda x: len(x)>9).reset_index(drop=True)
cf.tsne_non_trained_classes(model_c,emb_data,tnse_write_path,layer,max_len)

    
