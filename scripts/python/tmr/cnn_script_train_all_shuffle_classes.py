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
data_path='data/cluster_dataframes/'
model_save_path='data/models/shuffle/'
basename='swiss100'
expr_path='data/experiment/shuffle/'
fractions=np.linspace(0,1,21)
replicates=3 

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


df_expriment=pd.DataFrame()
#%%
#generate datasets for fitting
for r in replicates:
    for f in fractions:
        seq_df=cf.load_seq_dataframe(data_path)
        uniq_anno=seq_df.annotation.unique()
        num_classes=len(uniq_anno)
        annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
        seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation').drop(['Component_1','Component_2','Cluster','id','annotation'],axis=1)
        
        seq_df_shuffle=cf.randomize_groups(seq_df,'ydata',f)
        
        #%%
        #generate training data for annotation/cluster datasets
        ##annotations
        train_a=seq_df_shuffle.groupby(['ydata']).sample(n=sample_n)
        seq_df_shuffle=seq_df_shuffle.drop(train_a.index)
        train_a_one_hot=cf.seq_one_hot(train_a['sequence'],
                                      seq_type=seq_type,
                                      max_len=max_len,
                                      seq_resize=seq_resize)
        
        #%%
        #generate validation data for annotation/cluster datasets
        ##annotation
        validation_a=seq_df_shuffle.groupby(['ydata']).sample(n=sample_n)
        seq_df_shuffle=seq_df_shuffle.drop(validation_a.index)
        validation_a_one_hot=cf.seq_one_hot(validation_a['sequence'],
                                      seq_type=seq_type,
                                      max_len=max_len,
                                      seq_resize=seq_resize)
        
        #generate test data for annotation/cluster datasets
        ##annotation
        
        test_a=seq_df_shuffle
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
        
        
        model_a.save(model_save_path + str(r) + '_' + str(f) + '_' + basename + '.h5')
        
        results=model_a.evaluate(test_a_one_hot,ytest_a)
        
        df_expriment.append({'loss': results[0],
                             'accuracy': results[1],
                             'replicate': r,
                             'fraction': f,
                             'epoch': epochs},
                            ignore_index=True)

df_expriment.to_csv(expr_path+basename+'_'+'shuffle')

    
