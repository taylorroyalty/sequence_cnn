from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation, Flatten, Input, Multiply
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam

def original_blstm(num_classes, num_letters, sequence_length, embed_size=50):
	model = Sequential()
	model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
	model.add(Dropout(0.5))
	#model.add(LSTM(num_classes, activation="softmax", name="AV"))
	model.add(LSTM(embed_size, activation="tanh"))
	model.add(Dense(num_classes, activation=None, name="AV"))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
	return model

def dna_blstm(num_classes, num_letters, sequence_length, mask_length=None, embed_size=256):
	model = Sequential()
        model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=26, kernel_size=3, strides=3, padding="valid", activation="relu"))
        model.add(Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
        model.add(Dropout(0.5))
        #model.add(LSTM(num_classes, activation="softmax", name="AV"))
        model.add(LSTM(embed_size, activation="tanh"))
        model.add(Dense(num_classes, activation=None, name="AV"))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

def dna_mask_blstm(num_classes, num_letters, sequence_length, mask_length, embed_size=256):
	x = Input(shape=(sequence_length, num_letters))
	m = Input(shape=(mask_length, 1))
	conv1 = Conv1D(filters=26, kernel_size=3, strides=3, padding="valid", activation="relu")(x)
	conv2 = Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu")(conv1)
	pool = MaxPooling1D(pool_length=13, stride=13)(conv2)
	masked = Multiply()([pool, m])
	masking = Masking(mask_value=0)(masked)
	drop1 = Dropout(0.2)(masking)
	blstm = Bidirectional(LSTM(320, activation="tanh", return_sequences=True))(drop1)
	drop2 = Dropout(0.5)(blstm)
	lstm = LSTM(embed_size, activation="tanh")(drop2)
	dense = Dense(num_classes, activation=None, name="AV")(lstm)
	out = Activation("softmax")(dense)
	model = Model(inputs=[x,m], outputs=out)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
	return model

#returns an lstm model for amino acids from keras. Inputs [x,m] and output out. x, m, and out defined below
def aa_mask_blstm(num_classes, num_letters, sequence_length, mask_length, embed_size=256):
	x = Input(shape=(sequence_length, num_letters))
        m = Input(shape=(mask_length, 1))
        conv = Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu")(x)
        pool = MaxPooling1D(pool_length=13, stride=13)(conv)
        masked = Multiply()([pool, m])
        masking = Masking(mask_value=0)(masked)
        drop1 = Dropout(0.2)(masking)
        blstm = Bidirectional(LSTM(320, activation="tanh", return_sequences=True))(drop1)
        drop2 = Dropout(0.5)(blstm)
        lstm = LSTM(embed_size, activation="tanh")(drop2)
        dense = Dense(num_classes, activation=None, name="AV")(lstm)
        out = Activation("softmax")(dense)
        model = Model(inputs=[x,m], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

def dspace(num_classes, num_letters, sequence_length, embed_size=256):
	model = Sequential()
	model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=26, kernel_size=3, strides=3, padding="valid", activation="elu", name='dna2aa', kernel_initializer='he_uniform'))

	model.add(Conv1D(16, 3, activation='elu', padding='same', name='encoder_conv0', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn0'))
	model.add(Conv1D(24, 3, activation='elu', padding='same', name='encoder_conv1', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn1'))
	model.add(MaxPooling1D())

	model.add(Conv1D(32, 5, activation='elu', padding='same', name='encoder_conv2', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn2'))
	model.add(Conv1D(48, 5, activation='elu', padding='same', name='encoder_conv3', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn3'))

	model.add(Conv1D(64, 7, activation='elu', padding='same', name='encoder_conv4', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn4'))
	model.add(Conv1D(96, 7, activation='elu', padding='same', name='encoder_conv5', kernel_initializer='he_uniform'))
	model.add(BatchNormalization(name='encoder_bn5'))
	model.add(MaxPooling1D())

	model.add(Flatten())
	model.add(Dense(2048, activation='elu', kernel_initializer='he_uniform', name='encoder_aff0'))
	model.add(BatchNormalization(name='encoder_bn6'))
	model.add(Dense(1024, activation='elu', kernel_initializer='he_uniform', name='encoder_aff1'))
	model.add(BatchNormalization(name='encoder_bn7'))
	model.add(Dense(512, activation='elu', kernel_initializer='he_uniform', name='encoder_aff2'))
	model.add(BatchNormalization(name='encoder_bn8'))
	model.add(Dense(embed_size, activation='elu', kernel_initializer='he_uniform', name='encoder_aff3'))
	#obviously inaccurate name, to match embedding in previous models
	model.add(BatchNormalization(name='lstm_2'))

	model.add(Dense(num_classes, activation=None, name="AV"))
        model.add(Activation("softmax"))

	model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.001), metrics=['accuracy'])
        return model
