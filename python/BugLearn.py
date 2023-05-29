'''
Created on Jun 23, 2017

@author: Michael Pradel, Sabine Zach
'''

import sys
import json
from os.path import join
from os import getcwd
from collections import namedtuple
import math

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers.core import Dense, Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import time
import numpy as np
import Util
import LearningDataSwappedArgs
import LearningDataBinOperator
import LearningDataSwappedBinOperands
import LearningDataIncorrectBinaryOperand
import LearningDataIncorrectAssignment
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pattern", help="Kind of data to extract", choices=["SwappedArgs", "BinOperator", "SwappedBinOperands", "IncorrectBinaryOperand", "IncorrectAssignment"], required=True)
parser.add_argument(
    "--token_emb", help="JSON file with token embeddings", required=True)
parser.add_argument(
    "--type_emb", help="JSON file with type embeddings", required=True)
parser.add_argument(
    "--node_emb", help="JSON file with AST node embeddings", required=True)
parser.add_argument(
    "--training_data", help="JSON files with training data", required=True, nargs="+")
parser.add_argument(
    "--out", help="Target directory for trained model")


Anomaly = namedtuple("Anomaly", ["message", "score"])

#input-output pairs: (vpos , 0) and (vneg , 1).
def prepare_xy_pairs(gen_negatives, data_paths, learning_data):
    xs = []
    ys = []
    # keep calls in addition to encoding as x,y pairs (to report detected anomalies)
    code_pieces = []

    for code_piece in Util.DataReader(data_paths):
        learning_data.code_to_xy_pairs(gen_negatives, code_piece, xs, ys,
                                       name_to_vector, type_to_vector, node_type_to_vector, code_pieces)
    x_length = len(xs[0])

    print("Stats: " + str(learning_data.stats))
    print("Number of x,y pairs: " + str(len(xs)))
    print("Length of x vectors: " + str(x_length))
    xs = np.array(xs)
    ys = np.array(ys)
    return [xs, ys, code_pieces]


def sample_xy_pairs(xs, ys, number_buggy):
    sampled_xs = []
    sampled_ys = []
    buggy_indices = []
    for i, y in enumerate(ys):
        if y == [1]:
            buggy_indices.append(i)
    sampled_buggy_indices = set(np.random.choice(
        buggy_indices, size=number_buggy, replace=False))
    for i, x in enumerate(xs):
        y = ys[i]
        if y == [0] or i in sampled_buggy_indices:
            sampled_xs.append(x)
            sampled_ys.append(y)
    return sampled_xs, sampled_ys


if __name__ == '__main__':
    print("BugLearn started with " + str(sys.argv))
    time_start = time.time()

    args = parser.parse_args()
    pattern = args.pattern
    name_to_vector_file = args.token_emb
    type_to_vector_file = args.type_emb
    node_type_to_vector_file = args.node_emb
    training_data_paths = args.training_data
    out_dir = args.out
    if out_dir is None:
        time_stamp = math.floor(time.time() * 1000)
        "bug_detection_model_"+str(time_stamp)

    with open(name_to_vector_file) as f:
        name_to_vector = json.load(f)
    with open(type_to_vector_file) as f:
        type_to_vector = json.load(f)
    with open(node_type_to_vector_file) as f:
        node_type_to_vector = json.load(f)

    if pattern == "SwappedArgs":
        learning_data = LearningDataSwappedArgs.LearningData()
    elif pattern == "BinOperator":
        learning_data = LearningDataBinOperator.LearningData()
    elif pattern == "SwappedBinOperands":
        learning_data = LearningDataSwappedBinOperands.LearningData()
    elif pattern == "IncorrectBinaryOperand":
        learning_data = LearningDataIncorrectBinaryOperand.LearningData()
    elif pattern == "IncorrectAssignment":
        learning_data = LearningDataIncorrectAssignment.LearningData()
    else:
        raise Exception(f"Unexpected pattern: {pattern}")
    # elif pattern == "MissingArg":
    ##    learning_data = LearningDataMissingArg.LearningData()
    # not yet implemented

    print("Statistics on training data:")
    learning_data.pre_scan(training_data_paths, [])

    # prepare x,y pairs for learning without negative examples
    print("Preparing xy pairs for training data:")
    learning_data.resetStats()
    xs_training, ys_training, _ = prepare_xy_pairs(
        True, training_data_paths, learning_data)
    x_length = len(xs_training[0])
    print("Training examples   : " + str(len(xs_training)))
    print(learning_data.stats)



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
print(xs_training[:10])
print(ys_training[:10])
'''  
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

print(x_train[:10])
print(x_val[:10])

y_val[:5]

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
'''  
'''
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(x_length,))
embedding_layer = TokenAndPositionEmbedding(x_length, 10000, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(xs_training, ys_training, 
                    batch_size=64, epochs=2, 
                   )

model.save_weights("predict_class.h5")

#results = model.evaluate(x_val, y_val, verbose=2)

#for name, value in zip(model.metrics_names, results):
#    print("%s: %.3f" % (name, value))
    
'''

'''
    # create a model (simple feedforward network)
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(x_length,)))
    model.add(Dense(200, input_dim=x_length,
                    activation="relu", kernel_initializer='normal'))
    model.add(Dropout(0.2))
    #model.add(Dense(200, activation="relu"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer='normal'))

    # train model
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    
    history = model.fit(xs_training, ys_training,
                        batch_size=100, epochs=10, verbose=1)

    model.save(out_dir)

    time_learning_done = time.time()
    print("Time for learning (seconds): " +
          str(round(time_learning_done - time_start)))

    print(f"Bug detection model saved to {out_dir}")
'''
