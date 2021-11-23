# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:56:04 2020

@author: Peng
"""

#from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
#from sklearn.mixture import GaussianMixture
#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
#from random import sample
import sys
import pandas as pd
import numpy as np
import warnings

sys.path.insert(1,'scripts/python/tmr/')
import cnn_functions as cf

emb_data_swiss='data/swiss_data_variants/swiss50_embedding.tsv'
write_path='data/experiment/outlier_detection/optimal_eps_dbscan'
embed_size = 256
cores=20
eps_array=np.linspace(10,100,num=20)
#n_components=2

swiss_data=pd.read_csv(emb_data_swiss,sep='\t')
def parallel_outlier_detection(swiss_data,embed_size,eps):

	outlier_df=pd.DataFrame()
	uniq_anno=swiss_data.annotation.unique()
	for anno in uniq_anno:
		tmp_data=swiss_data[swiss_data.annotation.isin([anno])]
#			tmp_uniq_anno=uniq_anno[uniq_anno!=anno]
#			tmp_anno=str(sample(list(tmp_uniq_anno),1))
		embed=tmp_data[tmp_data.columns[-embed_size:]].to_numpy()

		scale_o=StandardScaler()
		embed=scale_o.fit_transform(embed)

#			tsne = TSNE(n_components=n_components,random_state=0)
#			embed_transformed = tsne.fit_transform(embed)
	
#			pca_o=PCA()
#			pca_transformed=pca_o.fit(embed)
#			cumsum=np.cumsum(pca_transformed.explained_variance_ratio_)
#			d=np.argmax(cumsum>=0.95)+1

#			pca_o=PCA(n_components=d)
#			embed_transformed=pca_o.fit_transform(embed)

		dbscan_o=DBSCAN(algorithm='kd_tree',eps=eps)
		dbscan_clust=dbscan_o.fit(embed)
#			gm=GaussianMixture(n_components=1)
#			gm.fit(embed_transformed)
#			density=gm.score_samples(embed_transformed)
#			density_thres=np.percentile(density,50)
#			anomolies=pca_transformed[density<density_thres]
		n_outlier=np.sum([dbscan_clust.labels_==-1])
		tmp=pd.DataFrame({'n': [len(tmp_data)], 'total_outliers': [n_outlier], 'annotation': [anno], 'eps': [eps]})
		outlier_df=outlier_df.append(tmp)

	outlier_df.to_csv(write_path + '_' + str(eps) + '_' + '.tsv',sep='\t')


print('start parallel')
Parallel(n_jobs=cores)(delayed(parallel_outlier_detection)(swiss_data,embed_size,eps) for eps in eps_array)





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

