
import random
import csv

from keras.models import load_model

from load_data import load_csv, get_onehot
from model_templates import dna_mask_blstm

model_name = 'blstm_mask_stride1_dna_100class_4500'
model_file = '../models/'+model_name+'.h5'
data_dir = '/mnt/data/computervision/dna_100class_train80_val10_test10'
sequence_length = 4500
num_classes = 100
num_letters = 4
#masking is required
mask_len = 344
model_template = dna_mask_blstm

random.seed(0)

#percent is an integer
def delete_segment(s, percent, align_3):
	cut_len = len(s) * percent / 100
	if align_3:
                cut_len = cut_len / 3 * 3
	start = random.randint(0, len(s) - cut_len)
	if align_3:
		start = start / 3 * 3
	return s[0:start] + s[start+cut_len:len(s)]

def substitute(s, percent):
	options = {'A':['C','G','T'],'C':['A','G','T'],'G':['A','C','T'],'T':['A','C','G']}
	n = len(s) * percent / 100
	indices = random.sample(xrange(len(s)), n)
	l = list(s)
	for i in indices:
		if l[i] in options:
			l[i] = random.choice(options[l[i]])
	return ''.join(l)





model = load_model(model_file)
model.summary()

results = []

for percent in range(2,22,2):
	#mode 0: substitute, mode 1: 3-aligned cut, mode 2: unaligned cut
	row = [percent]
	for mode in range(3):
		test_data = load_csv(data_dir + '/test.csv', divide=2)
		print len(test_data)
	
		for i in range(len(test_data)):
			(x, y) = test_data[i]
			if mode == 0:
				test_data[i] = (substitute(x, percent), y)
			else:
				test_data[i] = (delete_segment(x, percent, mode == 1), y)
			#if i % 100000 == 99999:
			#	print i+1

		test_x, test_y, test_m = get_onehot(test_data, None, is_dna_data=True, seq_len=sequence_length, num_classes=num_classes, mask_len=mask_len)
	
		acc =  model.evaluate([test_x, test_m], test_y, batch_size=100, verbose=1)[1]
		print percent, mode, acc

		row.append(acc)
		del test_data, test_x, test_y, test_m
	results.append(row)
	with open('../results/'+model_name+'_mutation_graphs.csv', 'w') as outfile:
		w = csv.writer(outfile)
		for row in results:
			w.writerow(row)
