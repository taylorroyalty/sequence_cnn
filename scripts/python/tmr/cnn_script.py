# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
#%%
# Libraries and modules
import pandas as pd
import random
import numpy as np
# from model_templates_tmr import original_blstm
from keras.utils import to_categorical

import sys

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf
#%%
#Inputs
layer="lstm_3"
tnse_write_path='data/tnse_results/'
emb_data_path='data/swiss_1_99.tsv'
cluster_dataframe_path='data/cluster_dataframes/'
model_save_path='data/models/'

magn=20
max_len=100
sample_frac=0.2
is_dna_data=False
mask_length = None
embed_size = 256
batch_size=100
epochs=50


#%%
#generate datasets for fitting
seq_df=cf.load_seq_dataframe(cluster_dataframe_path)
uniq_anno=seq_df.annotation.unique()
num_classes=len(uniq_anno)
annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation')
seq_cluster=seq_df.loc[seq_df['Cluster'] > -1]
seq_cluster_noise=seq_df.loc[seq_df['Cluster'] == -1]
seq_cluster_a=seq_cluster

#%%
#generate training data for annotation/cluster datasets
##annotations
# train_a=seq_df.groupby(['annotation']).sample(frac=sample_frac)
# seq_df=seq_df.drop(train_a.index)
# train_a_one_hot=aa_one_hot(train_a['sequence'],max_len=max_len)
train_a=seq_cluster_a.groupby(['annotation']).sample(frac=sample_frac)
seq_cluster_a=seq_cluster_a.drop(train_a.index)
train_a_one_hot=cf.aa_one_hot(train_a['sequence'],max_len=max_len)

##clusters
train_c=seq_cluster.groupby(['annotation','Cluster']).sample(frac=sample_frac)
seq_cluster=seq_cluster.drop(train_c.index)
train_c_one_hot=cf.aa_one_hot(train_c['sequence'],max_len=max_len)

#%%
#generate validation data for annotation/cluster datasets
##annotation
# validation_a=seq_df.groupby(['annotation']).sample(frac=sample_frac)
# seq_df=seq_df.drop(validation_a.index)
# validation_a_one_hot=aa_one_hot(validation_a['sequence'],max_len=max_len)
validation_a=seq_cluster_a.groupby(['annotation']).sample(frac=sample_frac)
seq_cluster_a=seq_cluster_a.drop(validation_a.index)
validation_a_one_hot=cf.aa_one_hot(validation_a['sequence'],max_len=max_len)

##clusters
validation_c=seq_cluster.groupby(['annotation','Cluster']).sample(frac=sample_frac)
seq_cluster=seq_cluster.drop(validation_c.index)
validation_c_one_hot=cf.aa_one_hot(validation_c['sequence'],max_len=max_len)

#generate test data for annotation/cluster datasets
##annotation

test_a=seq_cluster_a
test_a_one_hot=cf.aa_one_hot(test_a['sequence'],max_len=max_len)
#clusters
test_c=seq_cluster
test_c_one_hot=cf.aa_one_hot(test_c['sequence'],max_len=max_len)
test_noise_one_hot=cf.aa_one_hot(seq_cluster_noise['sequence'],max_len=max_len)

#%%
##generate y data for annotation/cluster datasets
##annotation
ytrain_a=to_categorical(np.array(train_a.ydata,dtype='uint8'),num_classes)
yvalidation_a=to_categorical(np.array(validation_a.ydata,dtype='uint8'),num_classes)
ytest_a=to_categorical(np.array(test_a.ydata,dtype='uint8'),num_classes)
#clusters
ytrain_c=to_categorical(np.array(train_c.ydata,dtype='uint8'),num_classes)
yvalidation_c=to_categorical(np.array(validation_c.ydata,dtype='uint8'),num_classes)
ytest_c=to_categorical(np.array(test_c.ydata,dtype='uint8'),num_classes)
ytest_noise=to_categorical(np.array(seq_cluster_noise.ydata,dtype='uint8'),num_classes)

num_letters = 4 if is_dna_data else 26
# sequence_length = train_one_hot.shape[1]
model_a= cf.original_blstm(num_classes, num_letters, max_len, embed_size=embed_size)
model_c= cf.original_blstm(num_classes, num_letters, max_len, embed_size=embed_size)

# n_train_a=train_a_one_hot.shape[0]
# n_train_c=train_c_one_hot.shape[0]

n_validation_a=validation_a_one_hot.shape[0]
n_validation_c=validation_c_one_hot.shape[0]

model_a.fit(x=train_a_one_hot,y=ytrain_a,batch_size=batch_size,
            validation_data=(validation_a_one_hot,yvalidation_a),validation_batch_size=magn*batch_size,
            epochs=epochs)
model_c.fit(x=train_c_one_hot,y=ytrain_c,batch_size=batch_size,
            validation_data=(validation_c_one_hot,yvalidation_c),validation_batch_size=magn*batch_size,
            epochs=epochs)


model_a.save(model_save_path + 'swiss100_annotation_only.h5')
model_c.save(model_save_path + 'swiss100_clusters.h5')

model_a.evaluate(test_a_one_hot,ytest_a)
model_c.evaluate(test_c_one_hot,ytest_c)
model_a.evaluate(test_noise_one_hot,ytest_noise)
model_c.evaluate(test_noise_one_hot,ytest_noise)

emb_data=pd.read_csv(emb_data_path,sep='\t').filter(lambda x: len(x)>9).reset_index(drop=True)
cf.tsne_non_trained_classes(model_a,emb_data,tnse_write_path,layer,max_len)

# for i in range(epochs):
#     #generate indices for building training batch datasets
#     sample_train_a_index=random.sample(range(n_train_a),batch_size)
#     sample_train_c_index=random.sample(range(n_train_c),batch_size)
    
#     #generate indices for building validation batch datasets
#     sample_validation_a_index=random.sample(range(n_validation_a),batch_size)
#     sample_validation_c_index=random.sample(range(n_validation_c),batch_size)
    
#     #batch datasets
#     ##training
#     tmp_train_a_x=train_a_one_hot[sample_train_a_index,:,:]
#     tmp_train_a_y=ytrain_a[sample_train_a_index,:]
#     tmp_train_c_x=train_c_one_hot[sample_train_c_index,:,:]
#     tmp_train_c_y=ytrain_c[sample_train_c_index,:]
    
#     ##validation
#     tmp_validation_a_x=validation_a_one_hot[sample_validation_a_index,:,:]
#     tmp_validation_a_y=yvalidation_a[sample_validation_a_index,:]
#     tmp_validation_c_x=validation_c_one_hot[sample_validation_c_index,:,:]
#     tmp_validation_c_y=yvalidation_c[sample_validation_c_index,:]
    
#     #fit models with batch datasets
#     model_a.train_on_batch(tmp_train_a_x,tmp_train_a_y,validation_data=(tmp_validation_a_x,tmp_validation_a_y))
#     model_c.train_on_batch(tmp_train_c_x,tmp_train_c_y,validation_data=(tmp_validation_c_x,tmp_validation_c_y)) 




# model.evaluate(test_one_hot,ytest)
# model.evaluate(test_noise_one_hot,ytest_noise)

##fit model

# model.fit(x=train_one_hot,y=ytrain,batch_size=100,epochs=500,validation_data=(validation_one_hot,yvalidation))
# =============================================================================
# model.evaluate(test_one_hot,ytest)
# 
# 
# 
# #cluster sampling
# train=seq_cluster.groupby(['annotation','Cluster']).sample(5)
# seq_cluster=seq_cluster.drop(train.index)
# validation=seq_cluster.groupby(['annotation','Cluster']).sample(5)
# seq_cluster=seq_cluster.drop(validation.index)
# test=seq_cluster
# 
# 
# 
# 
# 
# num_letters = 4 if is_dna_data else 26
# sequence_length = train_one_hot.shape[1]
# model = original_blstm(num_classes, num_letters, max_len, embed_size=embed_size)
# 
# ##fit model
# ytrain=to_categorical(np.array(train.ydata,dtype='uint8'),num_classes)
# yvalidation=to_categorical(np.array(validation.ydata,dtype='uint8'),num_classes)
# ytest=to_categorical(np.array(test.ydata,dtype='uint8'),num_classes)
# ytest_noise=to_categorical(np.array(seq_cluster_noise.ydata,dtype='uint8'),num_classes)
# =============================================================================

# # model.fit(x=train_one_hot,y=ytrain,batch_size=100,epochs=500,validation_data=(validation_one_hot,yvalidation))
# model.evaluate(test_one_hot,ytest)
# model.evaluate(test_noise_one_hot,ytest_noise)

# model.save(model_save_path + 'swiss100_annotation_clusters.h5')

    