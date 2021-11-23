import cnn_functions as cf
import pandas as pd
import sys
import numpy as np
import multiprocessing as mp

from keras.models import Model, load_model
from joblib import Parallel, delayed

max_len=300
embed_size = 256
batch_size=32
cnn_fun_path='scripts/python/tmr/'
seq_type='aa'
num_letters=26
sep=','
skip_first=0
ncpu=30
chunk_size=10000
seq_resize=False
#tara_path='data/swiss_data_variants/all_swiss_random.tsv'
tara_path='data/tara/sunagawa_500k_unique_transcripts.tsv'
model_path='data/models/iteration/iteration_swiss_n1/swiss_iteration_0.h5'
#write_path='data/swiss_data_variants/all_swiss_random_prediction.tsv'
write_path='data/drew_proposal/sunagawa_predictions.tsv'
sys.path.insert(1,'scripts/python/tmr/')
#pool=mp.Pool(ncpu)

df=pd.read_csv(tara_path,sep=sep)
model=load_model(model_path)


def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        if len(df[i*chunk_size:(i+1)*chunk_size])>0:
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

df_chunks=split_dataframe(df,chunk_size)
start=0
df['prediction']=None
for df_i in df_chunks:
	print(start)
	end=start+len(df_i)-1
	one_hot_seqs=cf.seq_one_hot(df_i['sequence'],seq_type=seq_type,
        	                               max_len=max_len,
                	                       seq_resize=seq_resize,
                        	               skip_first=skip_first)
	pred=model.predict(one_hot_seqs)
	df.loc[start:end,'prediction']=np.argmax(pred,axis=1)
	start=end+1


df.to_csv(write_path,sep='\t')


#df_chunks=df_chunks[0:30]
#results_o = [pool.apply_async(predict_tara_batch, args=(df_i, model, seq_type,max_len,seq_resize,skip_first)) for df_i in df_chunks]

#(print(df_i) for df_i in df_chunks)
#results = [r.get()[1] for r in results_o]

#pool.close()
#pool.join()
#[print(r.get(timeout=100)) for r in results_o]
