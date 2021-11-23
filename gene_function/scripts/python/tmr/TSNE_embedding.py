#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

swiss_path='data/drew_proposal/model_embedding.tsv'
tara_path='data/drew_proposal/sunagawa_500K_11182020_embedding.tsv'
write_path_swiss='data/tsne_results/swiss_random_iteration_0.tsv'
write_path_tara='data/tsne_results/tara_500K_swiss_random_iteration_0.tsv'
sep='\t'

swiss_data=pd.read_csv(swiss_path,sep=sep)
tara_data=pd.read_csv(tara_path,sep=sep)
n_components=2
embed_size=256

swiss_n=swiss_data.shape[0]

swiss_embed=swiss_data[swiss_data.columns[-embed_size:]].to_numpy()
tara_embed=tara_data[tara_data.columns[-embed_size:]].to_numpy()

#comb_embed=np.concatenate((swiss_embed,tara_embed),axis=0)

scale_o=StandardScaler()
scale_o.fit(swiss_embed)
swiss_embed=scale_o.transform(swiss_embed)
tara_embed=scale_o.transform(tara_embed)

#swiss_embed=combed_embed[0:swiss_n,:]
#tara_embed=combed_embed[-swiss_n:,:]

pca_o=PCA(n_components=n_components)
pca_o.fit(swiss_embed)
embed_s=pca_o.transform(swiss_embed)
embed_t=pca_o.transform(tara_embed)
print(pca_o.explained_variance_ratio_)


#tsne = TSNE(n_components=n_components,random_state=0)
#embed_t=tsne.fit_transform(swiss_embed)

data_o_swiss=pd.DataFrame({'comp1': embed_s[:,0],'comp2': embed_s[:,1]})
data_o_tara=pd.DataFrame({'comp1': embed_t[:,0],'comp2': embed_t[:,1]})
#print(type(swiss_data.annotation))
#print(type(tara_data.annotation))
#tmp=np.concatenate((swiss_data.annotation.to_numpy(),tara_data.annotation.to_numpy()))

data_o_swiss['class']=swiss_data['class'].to_numpy()
data_o_tara['id']=tara_data['id'].to_numpy()
#col_indx=embed_data.shape[1]-embed_size
#data_o=embed_data.iloc[:,:col_indx]
#data_o['component_1']=embed_t[:,0]
#data_o['component_2']=embed_t[:,1]

data_o_swiss.to_csv(write_path_swiss)
data_o_tara.to_csv(write_path_tara)

