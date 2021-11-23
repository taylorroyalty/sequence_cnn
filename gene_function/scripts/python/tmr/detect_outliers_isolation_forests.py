# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:56:04 2020

@author: Peng
"""

from sklearn.ensemble import IsolationForest
#from sklearn.cluster import DBSCAN
#from sklearn.mixture import GaussianMixture
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
#from random import sample
import sys
import pandas as pd
import numpy as np
import warnings

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

emb_data_swiss='data/swiss_data_variants/swiss50_embedding.tsv'
emb_data_tara='data/tara/sunagawa_500k_unique_transcripts_embedding.tsv'
write_path='data/experiment/outlier_detection/sunagawa_short_outlier'
embed_size = 256
cores=30
#n_components=2

swiss_data=pd.read_csv(emb_data_swiss,sep='\t')
tara_data=pd.read_csv(emb_data_tara,sep='\t')
#def parallel_outlier_detection(swiss_data,tara_data,tara_n,chunk_n,embed_size):
#	warnings.filterwarnings("ignore", category=RuntimeWarning) 
indx_max_save=tara_data.shape[1]-embed_size

#	def split_dataframe_size(df, chunk_size = 10000): 
#		chunks = list()
#		num_chunks = len(df) // chunk_size + 1
#		for i in range(num_chunks):
#			if i == np.max(range(num_chunks)):
#				chunks.append(df[i*chunk_size:])
#			else:
#				chunks.append(df[i*chunk_size:(i+1)*chunk_size])
#		return chunks

#	chunk_list=split_dataframe_size(tara_data,chunk_size=tara_n)
#	for chunk in chunk_list:
tara_embed=tara_data[tara_data.columns[-embed_size:]].to_numpy()

uniq_anno=swiss_data.annotation.unique()
def parallel_isolationForest(swiss_data,tara_embed,anno):
#for anno in uniq_anno:
#			tmp_uniq_anno=uniq_anno[uniq_anno!=anno]
#			tmp_anno=str(sample(list(tmp_uniq_anno),1))
	tmp_data=swiss_data[swiss_data.annotation.isin([anno])] # | (swiss_data.annotation==tmp_anno)]
	embed=tmp_data[tmp_data.columns[-embed_size:]].to_numpy()
#	embed=swiss_data[swiss_data.columns[-embed_size:]].to_numpy()
	isof_o=IsolationForest(random_state=0,contamination=0).fit(embed)
	isof_p=isof_o.predict(tara_embed)
#			embed=np.concatenate((embed,tara_embed),axis=0)

#			scale_o=StandardScaler()
#			embed=scale_o.fit_transform(embed)

#			tsne = TSNE(n_components=n_components,random_state=0)
#			embed_transformed = tsne.fit_transform(embed)
	
#			pca_o=PCA()
#			pca_transformed=pca_o.fit(embed)
#			cumsum=np.cumsum(pca_transformed.explained_variance_ratio_)
#			d=np.argmax(cumsum>=0.95)+1

#			pca_o=PCA(n_components=d)
#			embed_transformed=pca_o.fit_transform(embed)

#			dbscan_o=DBSCAN(algorithm='kd_tree',eps=38)
#			dbscan_clust=dbscan_o.fit(embed)
#			gm=GaussianMixture(n_components=1)
#			gm.fit(embed_transformed)
#			density=gm.score_samples(embed_transformed)
#			density_thres=np.percentile(density,50)
#			anomolies=pca_transformed[density<density_thres]
#			n_outlier=np.sum([dbscan_clust.labels_==-1])
#n_outlier=np.sum([isof_p==-1])
	tmp=pd.concat([tara_data[tara_data.columns[0:indx_max_save]],pd.DataFrame({'outlier': isof_p, 'annotation': anno})],axis=1)
	anno=str(anno).replace('/','_')
	anno=anno.replace('\\','_')
	anno=anno.replace(' ','_')
	tmp.to_csv(write_path+'_'+anno+'.tsv',sep='\t')
#	outlier_df.to_csv(write_path+'.tsv',sep='\t')


Parallel(n_jobs=cores)(delayed(parallel_isolationForest)(swiss_data,tara_embed,anno) for anno in uniq_anno)





#	tmp_data['comp1']=xx[:,0]
#	tmp_data['comp2']=xx[:,1]
#	tmp_data['comp3']=xx[:,2]

#	print('OPTICS...')
#	print(optics_clust.labels_)
#	print(np.min(tmp_data.cluster))

#emb_data['comp4']=xx[:,3]
#emb_data['comp5']=xx[:,4]
#print('Saving...')
#emb_data.to_csv(write_path,sep='\t')

