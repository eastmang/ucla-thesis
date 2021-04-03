#############################################################
#################### IMPORTING LIBRARIES ####################
#############################################################
import keras
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import pickle
from definitions.hyper_parameters import VOCAB_SIZE, MAX_LENGTH, EPO, BATCHES, VERBOSITY, TESTING_SIZE, RAND
from definitions.hyper_parameters import LEARNING_RATE, hidden_dim, nodes_lstm
from layers.attention_layers import Attention_add, Attention_local, Attention_mult

# setting the random seeds for reproducibility
np.random.seed(RAND)
random.seed(RAND)
tf.random.set_seed(RAND)

# setting an initializer for all of the layers for reproducibility
INITIALIZER = keras.initializers.glorot_uniform(seed=RAND)

#############################################################
#################### LOADING DATASETS #######################
#############################################################
os.chdir('D:\Grad 2nd year\Thesis\Thesis Project')
reviews = pickle.load(open("reviews", "rb"))
sentiment = pd.read_csv('sentiment.csv')

x_train, x_test, y_train, y_test = train_test_split(reviews, sentiment['sentiment'], random_state=RAND,
                                                    test_size=TESTING_SIZE)  # getting the training data with a set seed


#############################################################
#################### MAKING MODELS ##########################
#############################################################
def make_basic():
    input = Input(shape=(MAX_LENGTH, 1))
    x = Bidirectional(LSTM(nodes_lstm, input_shape=(MAX_LENGTH, 1), kernel_initializer=INITIALIZER))(input)
    x = Dense(1, activation='sigmoid', kernel_initializer=INITIALIZER)(x)
    model = Model(inputs=input, outputs=x)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def make_encoder_decoder():
    input = Input(shape=(MAX_LENGTH))
    x = Embedding(input_dim=VOCAB_SIZE + 1, output_dim=hidden_dim, input_length=MAX_LENGTH,
                  embeddings_initializer=INITIALIZER)(input)
    x = Bidirectional(LSTM(nodes_lstm, return_sequences=True, kernel_initializer=INITIALIZER))(x)
    x = Bidirectional(LSTM(nodes_lstm, kernel_initializer=INITIALIZER))(x)
    x = Dense(1, activation='sigmoid', kernel_initializer=INITIALIZER)(x)
    model = Model(inputs=input, outputs=x)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# This makes the intitial attention model. There is an attention type input that takes a string of either "local" or "add" and otherwise defaults to mult
def make_attention(attention_type):
    input = Input(shape=MAX_LENGTH)
    enc = Embedding(input_dim=VOCAB_SIZE + 1, output_dim=hidden_dim, input_length=MAX_LENGTH,
                    embeddings_initializer=INITIALIZER)(input)
    enc = Bidirectional(LSTM(nodes_lstm, return_sequences=True, kernel_initializer=INITIALIZER))(enc)
    dec = Bidirectional(LSTM(nodes_lstm, kernel_initializer=INITIALIZER))(enc)
    if attention_type == "local":
        att = Attention_local()([enc, dec])
    elif attention_type == "add":
        att = Attention_add()(enc)
    else:
        att = Attention_mult()([enc, dec])
    concat = Concatenate()([dec, att])
    output = Dense(1, activation='sigmoid', kernel_initializer=INITIALIZER)(concat)
    model = Model(inputs=input, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model_attention_local = make_attention("local")
model_attention_add = make_attention("add")
model_attention_mult = make_attention("mult")
model_basic = make_basic()
model_encoder_decoder = make_encoder_decoder()

#############################################################
################### RUNNING MODELS ##########################
#############################################################
model_attention_mult.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)

model_encoder_decoder.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)

model_attention_local.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)

model_attention_add.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)

model_basic.fit(x_train, y_train, epochs=EPO, batch_size=BATCHES, verbose=VERBOSITY)


#############################################################
################### GETTING RESULTS #########################
#############################################################
def result_calculator(mod):  # This function calculates the accuracy precision and recall of a given model
    y_pred = mod.predict(x_test).round().ravel()
    accuracy = accuracy_score(np.asarray(y_test), y_pred)
    precision = precision_score(np.asarray(y_test), y_pred)
    recall = recall_score(np.asarray(y_test), y_pred)
    return accuracy, precision, recall


accuracy_local, precision_local, recall_local = result_calculator(model_attention_local)

accuracy_mult, precision_mult, recall_mult = result_calculator(model_attention_mult)

accuracy_encoder_decoder, precision_encoder_decoder, recall_encoder_decoder = result_calculator(model_encoder_decoder)

accuracy_basic, precision_basic, recall_basic = result_calculator(model_basic)

accuracy_add, precision_add, recall_add = result_calculator(model_attention_add)

print("Local:", accuracy_local, precision_local, recall_local)
print("Mult:", accuracy_mult, precision_mult, recall_mult)
print("Add:", accuracy_add, precision_add, recall_add)
print("Encoder_Decoder:", accuracy_encoder_decoder, precision_encoder_decoder, recall_encoder_decoder)
print("Basic:", accuracy_basic, precision_basic, recall_basic)
