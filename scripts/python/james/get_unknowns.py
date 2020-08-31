import csv

is_dna_data = True

input_filename = '/mnt/data/sharing/nucleotide_annotation_data/all_annotation.tsv'
class_filename = '../results/dna_1000class_names.csv'
output_filename = '/mnt/data/computervision/dna_1000class_train80_val10_test10/unknowns.csv'
class_output_filename = '../results/dna_unknown_1000class_names.csv'
pair_output_filename = '../results/dna_unknown_1000class_pairs.csv'

data_size = 1200000 if is_dna_data else 800000

class_dict = dict()
with open(class_filename, 'r') as infile:
	r = csv.reader(infile)
	for row in r:
		class_dict[row[1]] = True
print class_dict

unknowns = []
unknown_class_dict = dict()

with open(input_filename, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        i = 0
        for row in reader:
                if i > 0: #ignore the first line
                        x = row[4 if is_dna_data else 3]
                        label = row[2]
			
                        if not label in class_dict:
                		unknowns.append((label, x))

				if not label in unknown_class_dict:
					unknown_class_dict[label] = 0
				unknown_class_dict[label] += 1

				if len(unknowns) == data_size:
					break
				
		i += 1
	print i

pair_dict = dict()
for (label, x) in unknowns:
	if unknown_class_dict[label] >= 2:
		if not label in pair_dict:
			pair_dict[label] = []
		p = pair_dict[label]
		if len(p) == 0 or (len(p) == 1 and x != p[0]):
			pair_dict[label].append(x)

with open(output_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for (label, x) in unknowns:
		#y is a single "unknown" class, equal to max class + 1
		w.writerow([len(class_dict), x])

with open(class_output_filename, 'w') as outfile:
	w = csv.writer(outfile)
	for key in unknown_class_dict:
		w.writerow([key, unknown_class_dict[key]])

with open(pair_output_filename, 'w') as outfile:
	w = csv.writer(outfile)
	i = 0
	for key in pair_dict:
		if len(pair_dict[key]) == 2:
			for x in pair_dict[key]:
				w.writerow([i, x])
			i += 1
print 'wrote ' + output_filename + ', ' + class_output_filename + ', ' + pair_output_filename

