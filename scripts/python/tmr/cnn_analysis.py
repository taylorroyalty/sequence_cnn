# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:38:52 2020

@author: Peng
"""

from plotnine import *
import pandas as pd
import os

def analyze_cluster_error():
    df_stat=pd.DataFrame()
    for f in os.listdir('data/experiment/cluster_error/'):
        df_tmp=pd.DataFrame()
        df_tmp2=pd.DataFrame()
        df=pd.read_csv('data/experiment/cluster_error/'+f)
        df=df.assign(correct=df.ydata==df.prediction)
        df_tmp[['accuracy','n']]=df.groupby(['Cluster','annotation']).correct.agg([lambda x: sum(x)/len(x), lambda x: len(x)])
        df_tmp2[['annotation','anno_acc']]=df.groupby(['annotation']).correct.agg([lambda x: sum(x)/len(x)]).reset_index()
        df_tmp=pd.merge(df_tmp,df_tmp2,on='annotation')
        df_tmp=df_tmp.assign(rank=df_tmp.accuracy.rank(method='max'))
        df_tmp['file']=f
        df_tmp['diff_acc']=df_tmp['anno_acc']-df_tmp['accuracy']
        df_tmp['rank']=df_tmp.diff_acc.rank(method='first',ascending=False)
        df_stat=df_stat.append(df_tmp)
        
    
    p = ggplot()+geom_point(df_stat,aes(x='rank',y='diff_acc',color='file'))#+geom_line(df_stat,aes(x='rank',y='tot_acc',color='file'))
    
    return p
                                                                                        
