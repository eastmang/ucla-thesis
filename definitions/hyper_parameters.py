#############################################################
#################### SETTING GLOBAL VARS ####################
#############################################################
# defining the global vocab size of 10000 for full doc 800 for one sentence
VOCAB_SIZE = 1200
# pad documents to a max length of 300 words if full doc and 100 for a sentence
MAX_LENGTH = 400
# number of epochs
EPO = 5
# number of batches
BATCHES = 250
# verbose setting
VERBOSITY = 1
# set the size of the testing group
TESTING_SIZE = .2
# set the random seed
RAND = 10
# set the width of the window for the local layer
WINDOW_WIDTH = 70
# learning rate for the models
LEARNING_RATE = .007
# setting the hidden dimensions
hidden_dim = 200
# setting the number of nodes in the LSTM layers
# the call is bidirectional so we will get twice the number of hidden layers
# therefore we want to divide by two to stay consistent
nodes_lstm = int(hidden_dim/2)