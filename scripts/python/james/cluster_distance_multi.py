import csv
import os
import numpy as np
from multiprocessing import Process
from sklearn.neighbors import KDTree
from keras.models import Model, load_model
from load_data import get_onehot

model_name = 'blstm_mask_embed64_aa_30class_1500'
input_filename = '/mnt/data/computervision/tara/TARA_AAsequence_site_region.tsv'

is_dna_data = False
seq_len = 1500
mask_len = 113

model_file = '../models/'+model_name+'.h5'
model = load_model(model_file)
embed_model = Model(inputs=model.input, outputs=model.get_layer("lstm_2").output)
embed_model.summary()

label_dict = dict()
rev_label_dict = dict()
sequence_dict = dict()
embed_dict = dict()

with open(input_filename) as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	i = 0
	for row in reader:
		if i > 0:
			x = row[0]
			label = row[2]
			if not label in label_dict:
				y = len(label_dict)
				label_dict[label] = y
				rev_label_dict[y] = label
				sequence_dict[y] = []
			y = label_dict[label]
			sequence_dict[y].append((x, y))
		i+=1
	print i

N = len(sequence_dict)
print 'done loading', N

for i in range(N):
	print len(sequence_dict[i])
	filename = '/mnt/data/computervision/tara/embed64/'+rev_label_dict[i]+'.npy'
	if os.path.exists(filename):
		embed_dict[i] = np.load(filename)
	else:
		x, y, m = get_onehot(sequence_dict[i], None, is_dna_data=is_dna_data, seq_len=seq_len, mask_len=mask_len)
		embed = embed_model.predict([x,m], batch_size=100, verbose=1)
		embed_dict[i] = embed
		del x, y, m
		np.save(filename, embed)
	del sequence_dict[i]
	print 'embedded', i, rev_label_dict[i]

	#embed_dict[i] = embed_dict[i][0:1000]

del sequence_dict, model, embed_model
result = []

tree_dict = dict()

for i in range(N):
	tree_dict[i] = KDTree(embed_dict[i], leaf_size=10)
	print 'tree', i

def distance(embed, tree, embed_name, tree_name):
	path = "/mnt/data/computervision/tara/results64/"
	dist_filename = path + embed_name + "_" + tree_name + "_distances.npy"
	ind_filename = path + embed_name + "_" + tree_name + "_indices.npy"
	if os.path.exists(dist_filename):
		dists = np.load(dist_filename)
		indices = np.load(ind_filename)
	else:
		(dists, indices) = tree.query(embed, k=1)
		np.save(dist_filename, dists)
		np.save(ind_filename, indices)
        dist = np.mean(dists)
        print embed_name, tree_name, dist

for i in range(N):
	for j in range(N):
		if not j == i:
			p = Process(target=distance, args=(embed_dict[i], tree_dict[j], rev_label_dict[i], rev_label_dict[j]))
			p.start()


