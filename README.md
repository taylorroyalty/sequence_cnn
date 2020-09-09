Authors: Taylor Royalty (scripts for preprocessing) and James Senter (cnn/rnn models)


base_model_script.py is a script which builds a cnn/rnn model with a specified sequence dataset. 

Notes:
--Descriptions for specific inputs are commented within base_model_script.py
--sequence datasets need to have one column specified as 'annotation' and 'sequence.'


cnn_functions.py contains the functions seq_one_hot, original_blstm, dna_blstm, and load_seq_dataframe.
Notes:
--seq_one_hot one-hot-encodes AA/DNA sequences
--original_blstm is a cnn/rnn model for AA sequences
--dna_blstm is a cnn/rnn model for DNA sequences
--load_seq_dataframe appends multiple files from a directory. This functionality is not used in base_model_script.py

The test_data folder contains 5 annotation classes, with 20 sequences each (total 100 sequences).

