import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(0, len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # find all unique characters in the text
    chars = sorted(list(set(text)))
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('-', ' ')
    text = text.replace('*', ' ')
    text = text.replace('/', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('à', ' ')
    text = text.replace('â', ' ')
    text = text.replace('è', ' ')
    text = text.replace('é', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    # remove as many non-english characters and character sequences as you can 
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    
    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
