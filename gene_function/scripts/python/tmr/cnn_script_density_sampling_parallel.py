# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:54:42 2020

@author: Peng
"""
#%%
# Libraries and modules
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf
from joblib import Parallel, delayed


#%%
#Inputs
cpu=30
data_path='data/cluster_dataframes/tsne_cluster_dataframes/'
dr_type='TSNE'
# all_save_path='data/density_sample/all_data'
tmp_save_path='data/density_sample/tmp/'
final_save_path='data/density_sample/KDE'
# n_thres=26


#Sampling Parameters
rep=3
val_sample=range(1,11,1) #start on 18
test_n=5 #test per cluster
# n_thres=26
# sample_rate=0.98

#%%
#generate datasets for fitting

seq_df=cf.load_seq_dataframe(data_path)
uniq_anno=seq_df.annotation.unique()
num_classes=len(uniq_anno)
annotation_ydata_df=pd.DataFrame({'ydata': range(num_classes),'annotation': uniq_anno})
seq_df=pd.merge(seq_df,annotation_ydata_df,on='annotation')
# seq_df=seq_df.groupby(['annotation','Cluster']).filter(lambda x: x['id'].count()>n_thres)
# n_sample=round(min(seq_df.groupby(['annotation'])['id'].count())*max_sample_rate)*2
# seq_df=seq_df.groupby(['annotation']).sample(n_sample)
seq_df=seq_df.reset_index(drop=True)
seq_df['o_index']=seq_df.index


for s in val_sample:
    train_sample=round((s-0.2*s)/0.2)
    for r in range(rep):
        def parallel_anno(r,s,dr_type,train_sample,seq_df,anno,tmp_save_path):
            print(anno,r,s)
            tmp=seq_df[seq_df.annotation == anno].reset_index(drop=True)         
            df_test=tmp.groupby(['annotation','Cluster']).sample(frac=0.15)
            df_test['replicate']=r
            df_test['n_sample']=train_sample
            df_test['dataset']='test'
            df_test['method']='all'
            df_test['s_index']=None
            
            tmp=tmp.drop(df_test.index,axis=0).reset_index()
            tmp['s_index']=tmp.index
            points=np.array(tmp[['Component_1','Component_2']])
            tmp_sample_rate=(train_sample+s)/points.shape[0]
            categories=np.array(tmp.Cluster+1,dtype='int32')
            df_sample=cf.sample_clusters(points,categories,tmp_sample_rate)
            df_sample=pd.merge(tmp,df_sample,on='s_index')
            df_sample['replicate']=r
            df_sample['n_sample']=train_sample
            df_train=df_sample.groupby('method').sample(frac=0.8)
            df_validation=df_sample.drop(df_train.index)
            df_train['dataset']='train'
            df_validation['dataset']='validation'
            
            df_fit=pd.DataFrame()
            df_fit=df_fit.append(df_train.drop('index',axis=1))
            df_fit=df_fit.append(df_validation.drop('index',axis=1))
            df_fit=df_fit.append(df_test)

            # seq_df.to_csv(all_save_path+'_'+dr_type+'_'+str(train_sample)+'.csv')
            anno=anno.replace('/','_')
            df_fit.to_csv(tmp_save_path+dr_type+'_'+anno+'_'+str(r)+'_'+str(train_sample)+'.csv')

        Parallel(n_jobs=cpu)(delayed(parallel_anno)(r,s,dr_type,train_sample,seq_df,anno,tmp_save_path) for anno in uniq_anno)
 
tmp=cf.load_seq_dataframe(tmp_save_path)

# for f in os.listdir(tmp_save_path):
#     os.remove(tmp_save_path+f)
    
tmp.to_csv(final_save_path+'_'+dr_type+'.csv')
