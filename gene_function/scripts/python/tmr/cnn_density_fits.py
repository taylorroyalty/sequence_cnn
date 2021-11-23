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
# all_path='data/density_sample/all_data.csv'
data_path='data/density_sample/train_val_TSNE.csv'
model_save_path='data/models/'
dr_type='TSNE'
save_test='data/experiment/density_sampling/TSNE_8_40'



#nn parameters
max_len=300
embed_size = 256
batch_size=32
epochs=50
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
seq_resize=False 

#%%
#generate datasets for fitting
# if new_model == True:
# test_all=pd.read_csv(all_path)
print('starting to load data')
data=pd.read_csv(data_path)
print('data loaded')
# seq_df=cf.load_seq_dataframe(data_path)
uniq_anno=data.annotation.unique()
num_classes=len(uniq_anno)
# annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
# seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation')
# seq_df=seq_df.groupby(['annotation','Cluster']).filter(lambda x: x['id'].count()>n_thres)
# # seq_cluster=seq_df.loc[seq_df['Cluster'] > -1]
# # seq_cluster_noise=seq_df.loc[seq_df['Cluster'] == -1]
# seq_cluster_a=seq_df
# seq_cluster=seq_df

method_u=data.method.unique()
replicate_u=data.replicate.unique()
sample_u=data.n_sample.unique()

df_results=pd.DataFrame()
for s in sample_u:
    for r in replicate_u:
        for m in method_u:
            print(s,r,m)
            if m == "all":
                continue
            train=data.loc[(data['method']==m) & 
                             (data['dataset']=="train") & 
                             (data['replicate']==r) & 
                             (data['n_sample']==s)].reset_index(drop=True)
            validation=data.loc[(data['method']==m) & 
                                 (data['dataset']=="validation") & 
                                 (data['replicate']==r) & 
                                 (data['n_sample']==s)].reset_index(drop=True)
            test=data.loc[(data['dataset']=="test") & 
                           (data['replicate']==r) & 
                           (data['n_sample']==s)].reset_index(drop=True)
            # index_drop=train.o_index
            # index_drop.append(validation.o_index)
            # test=test_all.drop(index_drop.tolist())
            
            
            train_one_hot=cf.seq_one_hot(train['sequence'],
                                       seq_type=seq_type,
                                       max_len=max_len,
                                       seq_resize=seq_resize)
            
            validation_one_hot=cf.seq_one_hot(validation['sequence'],
                                       seq_type=seq_type,
                                       max_len=max_len,
                                       seq_resize=seq_resize)
            
            test_one_hot=cf.seq_one_hot(test['sequence'],
                                       seq_type=seq_type,
                                       max_len=max_len,
                                       seq_resize=seq_resize)
            
            ytrain=to_categorical(np.array(train.ydata,dtype='uint32'),num_classes)
            yvalidation=to_categorical(np.array(validation.ydata,dtype='uint32'),num_classes)
            ytest=to_categorical(np.array(test.ydata,dtype='uint32'),num_classes)
            
            model= cf.original_blstm(num_classes,
                                    num_letters,
                                    max_len,
                                    embed_size=embed_size)
            
            n_validation=validation_one_hot.shape[0]
            model.fit(x=train_one_hot,y=ytrain,batch_size=batch_size,
                     validation_data=(validation_one_hot,yvalidation),
                     epochs=epochs)
            
            model.save(model_save_path + '_' + m + '_' + str(int(r)) + '_' + str(int(s)) + '_' + dr_type + '.h5')
             
            pred=model.predict(test_one_hot)
            pred=np.argmax(pred,axis=1)
            label=np.argmax(ytest,axis=1)
            print(sum(pred==label)/len(pred))
            test['prediction']=pred
            df_results=df_results.append(test)
            

df_results.to_csv(save_test + '.csv')
             


#%%
#generate training data for annotation/cluster datasets
##annotations
# train_a=seq_cluster_a.groupby(['annotation']).sample(n=sample_n)
# seq_cluster_a=seq_cluster_a.drop(train_a.index)
# train_a_one_hot=cf.seq_one_hot(train_a['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)

##clusters
# train_c=seq_cluster.groupby(['annotation','Cluster']).sample(n=sample_n)
# seq_cluster=seq_cluster.drop(train_c.index)
# train_c_one_hot=cf.seq_one_hot(train_c['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)

# #%%
# #generate validation data for annotation/cluster datasets
# ##annotation
# validation_a=seq_cluster_a.groupby(['annotation']).sample(n=sample_n)
# seq_cluster_a=seq_cluster_a.drop(validation_a.index)
# validation_a_one_hot=cf.seq_one_hot(validation_a['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)

# ##clusters
# validation_c=seq_cluster.groupby(['annotation','Cluster']).sample(n=sample_n)
# seq_cluster=seq_cluster.drop(validation_c.index)
# validation_c_one_hot=cf.seq_one_hot(validation_c['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)

# #generate test data for annotation/cluster datasets
# ##annotation

# test_a=seq_cluster_a
# test_a_one_hot=cf.seq_one_hot(test_a['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)
# #clusters
# test_c=seq_cluster
# test_c_one_hot=cf.seq_one_hot(test_c['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)
# test_noise_one_hot=cf.seq_one_hot(seq_cluster_noise['sequence'],
#                               seq_type=seq_type,
#                               max_len=max_len,
#                               seq_resize=seq_resize)

#%%
##generate y data for annotation/cluster datasets
# ##annotation
# ytrain_a=to_categorical(np.array(train_a.ydata,dtype='uint32'),num_classes)
# yvalidation_a=to_categorical(np.array(validation_a.ydata,dtype='uint32'),num_classes)
# ytest_a=to_categorical(np.array(test_a.ydata,dtype='uint32'),num_classes)
# #clusters
# ytrain_c=to_categorical(np.array(train_c.ydata,dtype='uint32'),num_classes)
# yvalidation_c=to_categorical(np.array(validation_c.ydata,dtype='uint32'),num_classes)
# ytest_c=to_categorical(np.array(test_c.ydata,dtype='uint32'),num_classes)
# ytest_noise=to_categorical(np.array(seq_cluster_noise.ydata,dtype='uint8'),num_classes)


# sequence_length = train_one_hot.shape[1]
# model_a= cf.original_blstm(num_classes,
#                            num_letters,
#                            max_len,
#                            embed_size=embed_size)
# model_c= cf.original_blstm(num_classes,
#                            num_letters,
#                            max_len,
#                            embed_size=embed_size)

# n_train_a=train_a_one_hot.shape[0]
# n_train_c=train_c_one_hot.shape[0]

# n_validation_a=validation_a_one_hot.shape[0]
# n_validation_c=validation_c_one_hot.shape[0]

# model_a.fit(x=train_a_one_hot,y=ytrain_a,batch_size=batch_size,
#             validation_data=(validation_a_one_hot,yvalidation_a),
#             epochs=epochs)
# model_c.fit(x=train_c_one_hot,y=ytrain_c,batch_size=batch_size,
#             validation_data=(validation_c_one_hot,yvalidation_c),
#             epochs=epochs)


# model_a.save(model_save_path + 'swiss100_annotation_only.h5')
# model_c.save(model_save_path + 'swiss100_clusters.h5')

# tmp=model_a.evaluate(test_a_one_hot,ytest_a)
# tmp2=model_a.predict_classes(test_a_one_hot)
# i_max=[uniq_anno[np.where(tmp2[i,:]==tmp2[i,:].max())] == test_a.annotation.iloc[i] for i in range(tmp2.shape[0])]
# sum(i_max)/len(i_max)    

# model_c.evaluate(test_c_one_hot,ytest_c)

# test_a['prediction']=model_a.predict(test_a_one_hot)
# test_c['prediction']=model_c.predict(test_c_one_hot)
# test_a['actual']=ytest_a
# test_c['actual']=ytest_c

# test_a.to_csv('data/experiment/cluster_error/'+anno_ex)
# test_c.to_csv('data/experiment/cluster_error/'+clust_ex)
# model_a.evaluate(test_noise_one_hot,ytest_noise)
# model_c.evaluate(test_noise_one_hot,ytest_noise)

# else:
#     from keras.models import load_model
#     model_c=load_model('data/models/swiss100_clusters.h5')

# if cluster_nontrain==True:
#     emb_data=pd.read_csv(emb_data_path,sep='\t').groupby("annotation").filter(lambda x: len(x)>9).reset_index(drop=True)
#     cf.tsne_non_trained_classes(model_c,emb_data,tnse_write_path,layer,max_len)


