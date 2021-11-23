# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 08:46:57 2020

@author: Peng
"""
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import sys
import numpy as np

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

skip_first=0
max_len=300
embed_size = 256
batch_size=32
epochs=4
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
#n_thres=26
seq_resize=False

model_save_path='data/models/swiss/'
model_name='iteration_3' 

df=pd.read_csv('data/tara/converge_novel_sequences/iteration_3/iteration_3.tsv',sep='\t')

uniq_anno=df.annotation.unique()
num_classes=len(uniq_anno)
anno_categorical=pd.DataFrame({'ydata': range(num_classes), 'annotation': uniq_anno})
df=pd.merge(df,anno_categorical,on='annotation')

if skip_first == 1:
    max_len-=1

one_hot_seqs=cf.seq_one_hot(df['sequence'],seq_type=seq_type,
                                       max_len=max_len,
                                       seq_resize=seq_resize,
                                       skip_first=skip_first)

y_cat=to_categorical(np.array(df.ydata,dtype='uint32'),num_classes)

X_train,X_test,y_train,y_test=train_test_split(one_hot_seqs,y_cat,
                                                  stratify=y_cat,
                                                  train_size=0.85)

X_train,X_validation,y_train,y_validation=train_test_split(one_hot_seqs,y_cat,
                                                              stratify=y_cat,
                                                              train_size=0.70/0.85)

model=cf.original_blstm(num_classes,
                            num_letters,
                            max_len,
                            embed_size=embed_size)

model.fit(x=X_train,y=y_train,batch_size=batch_size,
            validation_data=(X_validation,y_validation),
            epochs=epochs)

model.evaluate(X_test,y_test)
model.save(model_save_path + model_name + '.h5')
