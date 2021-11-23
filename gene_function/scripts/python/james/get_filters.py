import csv
from keras.models import load_model

model_name = 'blstm'

model = load_model('../models/'+model_name+'.h5')
model.summary()

weights = model.get_layer('conv1d_1').get_weights()[0]
print weights.shape

output_name = '../results/filters_'+model_name+'.csv'
with open(output_name, 'w') as outfile:
	w = csv.writer(outfile)
	for i in range(weights.shape[2]):
		for j in range(weights.shape[1]):
			w.writerow(weights[:,j,i].tolist())
		w.writerow([])

print 'wrote ' + output_name
