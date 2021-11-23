import csv
import numpy as np	

is_dna_data = True

num_classes = 1000
max_per_class = 20000

input_filename = '/mnt/data/sharing/nucleotide_annotation_data/all_annotation.tsv'
output_dir = '/mnt/data/computervision/dna_1000class_train80_val10_test10'
label_filename = '../results/dna_1000class_names.csv'

train_filename = output_dir + '/train.csv'
val_filename = output_dir + '/validation.csv'
test_filename = output_dir + '/test.csv'

class_sizes = dict()

with open(input_filename) as tsvfile:    
        reader = csv.reader(tsvfile, delimiter='\t')
        i = 0
        for row in reader:
                if i > 0: #ignore the first line
                        x = row[4] if is_dna_data else row[3]
                 	label = row[2]
			if not label in class_sizes:
				class_sizes[label] = 0
			class_sizes[label] += 1
                i += 1
		if i % 10000000 == 0:
			print i
        print 'total examples: ', i-1

top_classes = sorted(class_sizes, key=class_sizes.get)[-num_classes:]

label_mapping = dict()
labels = []
split_data = dict()

print 'top class sizes:'
for label in top_classes:
	label_mapping[label] = len(label_mapping)
	labels.append(label)
	split_data[label_mapping[label]] = []
	print class_sizes[label]

with open(input_filename) as tsvfile:    
	reader = csv.reader(tsvfile, delimiter='\t')
	i = 0
	for row in reader:
		if i > 0: #ignore the first line
			x = row[4] if is_dna_data else row[3]
			label = row[2]
			if label in label_mapping:
				y = label_mapping[label]
				if len(split_data[y]) < max_per_class:
					split_data[y].append(x)
		i += 1
		if i % 10000000 == 0:
			print i
	print 'chosen examples: ', i-1

with open(label_filename, 'w') as label_file:
	w = csv.writer(label_file)
	for i in range(len(labels)):
		w.writerow([i, labels[i]])
"""
print 'class sizes:'
for y in split_data:
	print len(split_data[y])
"""
with open(train_filename, 'w') as train_file, open(val_filename, 'w') as val_file, open(test_filename, 'w') as test_file:
	train_writer = csv.writer(train_file)
	val_writer = csv.writer(val_file)
	test_writer = csv.writer(test_file)

	for y in split_data:
		arr = split_data[y]
		l = len(arr)
		for i in range(l):
			x = arr[i]
			#80% train, 10% val, 10% test
			if i < l * 8 / 10:
				train_writer.writerow([y, x])
			elif i < l * 9 / 10:
				val_writer.writerow([y, x])
			else:
				test_writer.writerow([y, x])
