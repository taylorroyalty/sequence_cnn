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

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

#%%
#Inputs
data_path='data/cluster_dataframes/tsne_cluster_dataframes/'
dr_type='TSNE'
# all_save_path='data/density_sample/all_data'
train_val_save_path='data/density_sample/train_val'
# n_thres=26


#Sampling Parameters
rep=3
train_sample=10
test_val_sample=20
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

df_fit=pd.DataFrame()
for r in range(rep):
    for anno in uniq_anno:
        tmp=seq_df[seq_df.annotation == anno].reset_index(drop=True)
        tmp['s_index']=tmp.index
        points=np.array(tmp[['Component_1','Component_2']])
        tmp_sample_rate=train_sample/points.shape[0]
        categories=np.array(tmp.Cluster+1,dtype='int32')
        df_train=cf.sample_clusters(points,categories*0,tmp_sample_rate)
        df_train=pd.merge(tmp,df_train,on='s_index')
        df_train['dataset']='train'
        # df_tmp.dataset.iloc[df_tmp.groupby(['method']).sample(frac=0.5).index]='validation'
        df_train=df_train.drop('s_index',axis=1)
        df_train['replicate']=r
        
        tmp_sample_rate=train_sample/points.shape[0]*2
        df_test_val=cf.sample_clusters(points,categories*0,tmp_sample_rate)
        df_test_val=pd.merge(tmp,df_test_val,on='s_index')
        df_test_val['dataset']='test'
        df_test_val.dataset.iloc[df_test_val.groupby(['method']).sample(frac=0.5).index]='validation'
        df_test_val=df_test_val.drop('s_index',axis=1) 
        df_test_val['replicate']=r
        
        
        df_fit=df_fit.append(df_train)
        df_fit=df_fit.append(df_test_val)

# seq_df.to_csv(all_save_path+'_'+dr_type+'_'+str(train_sample)+'.csv')
df_fit.to_csv(train_val_save_path+'_'+dr_type+'_'+str(train_sample)+'.csv')

    
    
