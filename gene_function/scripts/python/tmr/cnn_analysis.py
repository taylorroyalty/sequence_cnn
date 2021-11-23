# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:38:52 2020

@author: Peng
"""

from plotnine import *
import pandas as pd
import os
import warnings

warnings.filterwarnings(action='ignore')

def analyze_cluster_error(d):
    df_stat=pd.DataFrame()
    for f in os.listdir(d):
        print(f)
        df_tmp2=pd.DataFrame()
        df=pd.read_csv(d+f)
        df=df.assign(correct=df.ydata==df.prediction)
        df['noise']=0
        df.noise[df.Cluster==-1]=1

        df_tmp=df.groupby(['Cluster','annotation','fraction','dr_type','noise']).correct.agg(accuracy = lambda x: sum(x)/len(x))
        df_tmp['n']=df.groupby(['Cluster','annotation','fraction','dr_type','noise']).correct.agg(lambda x: len(x))
        df_tmp=df_tmp.reset_index(drop=False)
   
        df_tmp2=df.groupby('annotation').correct.agg(anno_acc = lambda x: sum(x)/len(x)).reset_index(drop=False)
        df_tmp=pd.merge(df_tmp,df_tmp2,on='annotation')
        
        df_tmp2=df[df.noise==1].groupby('annotation').correct.agg(noise_acc = lambda x: sum(x)/len(x)).reset_index(drop=False)
        df_tmp=pd.merge(df_tmp,df_tmp2,on=['annotation'],how='outer')
        
        df_tmp2=df[df.noise==0].groupby('annotation').correct.agg(cluster_acc = lambda x: sum(x)/len(x)).reset_index(drop=False)
        df_tmp=pd.merge(df_tmp,df_tmp2,on=['annotation'])
        
        df_tmp['clust_noise_diff']=df_tmp.cluster_acc-df_tmp.noise_acc
        
        df_tmp=df_tmp.assign(rank=df_tmp.accuracy.rank(method='max'))
        # df_tmp['tot_acc']=sum(df.correct)/len(df.correct)
        df_tmp['method']=f.split("_")[0]
        df_tmp['samples']=float(f.split("_")[2].split(".")[0])
        df_tmp['diff_acc']=df_tmp['accuracy']-df_tmp['anno_acc']
        df_tmp['rank']=df_tmp.diff_acc.rank(method='first',ascending=False)
        df_tmp['rank2']=df_tmp.accuracy.rank(method='first',ascending=False)
        
        

        df_stat=df_stat.append(df_tmp)
        

    df_noise=df_stat.groupby(['noise','annotation','method','samples']).clust_noise_diff.agg('mean').reset_index(drop=False)    
    df_noise=df_stat.groupby(['noise','annotation','method','samples','clust_noise_diff']).cluster_acc.agg('mean').reset_index(drop=False)
    df_noise=df_stat.groupby(['noise','annotation','method','samples','clust_noise_diff','cluster_acc']).noise_acc.agg('mean').reset_index(drop=False)
    
    df_corr=pd.melt(df_noise.drop(['noise','clust_noise_diff','annotation'],axis=1),id_vars=['method','samples'],value_name="accuracy",var_name="data_type")
    df_corr2=df_corr.groupby(['samples','data_type','method']).accuracy.agg(m_acc='mean',sd_acc='std').reset_index(drop=False)
    # df_corr2['acc_std']=df_corr.groupby(['samples','data_type']).accuracy.agg('std')
    
    
    p1 = ggplot()+geom_line(df_stat,aes(x='rank2',y='accuracy',color='method'))+facet_wrap('samples')#+geom_line(df_stat,aes(x='rank',y='tot_acc',color='file'))
    p2 = ggplot()+geom_line(df_stat,aes(x='rank',y='diff_acc',color='method'))+facet_wrap('samples')
    p3 = ggplot()+geom_boxplot(df_noise,aes(x='factor(samples)',y='clust_noise_diff'))+facet_wrap('method')
    # p4 = ggplot()+geom_smooth(df_corr2,aes(x='samples',y='m_acc',color='data_type'),method='lm')+geom_point(df_corr2,aes(x='samples',y='m_acc',color='data_type'))+facet_wrap('method')
    p4 = ggplot()+geom_errorbar(df_corr2,aes(x='samples',ymax='m_acc+sd_acc',ymin='m_acc-sd_acc',color='data_type'))+geom_point(df_corr2,aes(x='samples',y='m_acc',color='data_type'))+facet_wrap('method')

    return p1,p2,p3,p4
                                                                                        
