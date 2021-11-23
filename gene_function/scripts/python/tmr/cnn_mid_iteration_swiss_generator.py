# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:54:26 2020

@author: Peng

"""
from keras.callbacks import EarlyStopping
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd
import numpy as np
import sys

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf   

skip_first=0
max_len=300
embed_size = 256
batch_size=32
epochs=10
chunk_size=10000
seq_type='aa'
num_letters=26
shuffle=True
#n_thres=26
seq_resize=False
model_save_path='data/models/iteration/'
model_name='swiss_iteration'
tara_path='data/tara/sunagawa_all_unique_transcripts_iteration_correct.tsv'
swiss_path='data/swiss_data_variants/swiss_n100_iteration.tsv'
start_iteration_path='data/iteration/swiss_tara_iteration_0.tsv'
write_path='data/iteration/swiss_tara_iteration'
i=0

callback=EarlyStopping(monitor='loss',patience=2)

#%%
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, sequences, y_cat, seq_type, max_len,seq_resize,
                 skip_first,batch_size=32,n_classes=2, shuffle=True):
        'Initialization'
        self.sequences=sequences
        self.y_cat=y_cat
        self.seq_resize=seq_resize
        self.seq_type=seq_type
        self.skip_first=skip_first
        self.max_len=max_len
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        sequences_batch = self.sequences.iloc[indexes]

        # Generate data
        X = self.__data_generation(sequences_batch)
        y = self.y_cat[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sequences):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X=one_hot_seqs=cf.seq_one_hot(sequences,seq_type=self.seq_type,
                            max_len=self.max_len,
                            seq_resize=self.seq_resize,
                            skip_first=self.skip_first)
         
        return X
    
def split_dataframe(df, chunk_size = 10000): 
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            if len(df[i*chunk_size:(i+1)*chunk_size])>0:
                chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks
#%%
df_cat=pd.read_csv(start_iteration_path,sep='\t')
df_cat=df_cat[df_cat['prediction']==1]
df_swiss=pd.read_csv(swiss_path,sep='\t')

delta_n=2
old_n=1 #value greater than 0
swiss_accuracy=[]
while delta_n>1.05:
#    if len(df_cat)>len(df_swiss):
#        df_cat=df_cat.sample(n=len(df_swiss))
    df=pd.concat([df_swiss,df_cat],axis=0)
    df.to_csv('data/test_for_script.tsv',sep='\t')
    uniq_anno=df['type'].unique()
    num_classes=len(uniq_anno)
    anno_categorical=pd.DataFrame({'ydata': range(num_classes), 'type': uniq_anno})
    df=pd.merge(df,anno_categorical,on='type')
    
    
    y_arr=np.array(df.ydata,dtype='uint32')
    x_arr=np.array(df.index,dtype='uint32')
    
    X_train_index,X_validation_index,y_train_index,y_validation_index=train_test_split(x_arr,
                                                   y_arr,
                                                   stratify=y_arr,
                                                   train_size=0.85)
    
    #X_train_index,X_validation_index,y_train_index,y_validation_index=train_test_split(X_train_index,
    #                                                           y_train_index,
    #                                                           stratify=y_train_index,
    #                                                           train_size=0.70/0.85) 
    


    X_train=df.iloc[X_train_index].sequence
    y_train=to_categorical(np.array(df.iloc[X_train_index].ydata,dtype='uint32'),num_classes)
    X_validation=df.iloc[X_validation_index].sequence
    y_validation=to_categorical(np.array(df.iloc[X_validation_index].ydata,dtype='uint32'),num_classes)
    
    swiss_cat=df.ydata[0]
    print(swiss_cat)
    
    #X_test=df.iloc[X_train_index].sequence
    
    training_generator = DataGenerator(X_train, y_train, seq_type,max_len,seq_resize,
                     skip_first,batch_size,num_classes,shuffle)
    validation_generator = DataGenerator(X_validation, y_validation, seq_type,max_len,seq_resize,
                     skip_first,batch_size,num_classes,shuffle)
    
    model=cf.original_blstm(num_classes,
                            num_letters,
                            max_len,
                            embed_size=embed_size)
    
    model.fit(x=training_generator,
                        epochs=epochs,
                        workers=20,
                        max_queue_size=20,
                        use_multiprocessing=True,
                        callbacks=[callback],
                        validation_data=validation_generator)
    
    del X_train, y_train, X_validation, y_validation, df
    
    model.save(model_save_path + model_name + '_' + str(i) + '.h5')  
    
    #Check swiss prot accuracy
    one_hot_seqs=cf.seq_one_hot(df_swiss['sequence'],seq_type=seq_type,
       	                               max_len=max_len,
               	                       seq_resize=seq_resize,
                       	               skip_first=skip_first)
    pred=model.predict(one_hot_seqs)
    swiss_accuracy.append(sum(np.argmax(pred,axis=1)==swiss_cat)/len(pred))
    print(swiss_accuracy)
    
    ###Predict Tara
    df_tara=pd.read_csv(tara_path,sep='\t')
    
    
    df_tara['prediction']=None
    df_tara['type']='not_swiss'
    df_chunks=split_dataframe(df_tara,chunk_size)
    start=0

    for df_i in df_chunks:
    	end=start+len(df_i)-1
    	one_hot_seqs=cf.seq_one_hot(df_i['sequence'],seq_type=seq_type,
            	                               max_len=max_len,
                    	                       seq_resize=seq_resize,
                            	               skip_first=skip_first)
    	pred=model.predict(one_hot_seqs)
    	df_tara.loc[start:end,'prediction']=np.argmax(pred,axis=1)
    	start=end+1
    
    
    df_tara.to_csv(write_path+'_'+str(i)+'.tsv',sep='\t')

    #Redefine novel group
    df_cat=df_tara[~(df_tara.prediction==swiss_cat)]
#    df_cat.to_csv('data/test_output.tsv',sep='\t')
    df_cat.drop(['prediction'],axis=1)
    delta_n=len(df_cat)/old_n
    old_n=len(df_cat)
    print(delta_n,old_n,len(df_cat))
    del df_tara, model, df_chunks
    i+=1

