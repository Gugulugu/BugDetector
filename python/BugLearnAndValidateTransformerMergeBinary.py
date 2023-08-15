'''
Created on Jun 23, 2017

@author: Michael Pradel, Sabine Zach
'''

import sys
import json
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math
import argparse
import os
import gc
import random
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers.core import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense, Flatten
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




import time
import numpy as np
import Util
import LearningDataSwappedArgs
import LearningDataBinOperator
import LearningDataSwappedBinOperands
import LearningDataIncorrectBinaryOperand
import LearningDataIncorrectAssignment
from Preprocessing import PreprocessingData



parser = argparse.ArgumentParser()
parser.add_argument(
    "--token_emb", help="JSON file with token embeddings", required=True)
parser.add_argument(
    "--training_data_Swapped", help="JSON files with training data", required=True, nargs="+")
parser.add_argument(
    "--training_data_BinOp", help="JSON files with training data", required=True, nargs="+")
parser.add_argument(
    "--training_data_IncBinOp", help="JSON files with training data", required=True, nargs="+")
parser.add_argument(
    "--validation_data_Swapped", help="JSON files with validation data", required=True, nargs="+")
parser.add_argument(
    "--validation_data_BinOp", help="JSON files with validation data", required=True, nargs="+")
parser.add_argument(
    "--validation_data_IncBinOp", help="JSON files with validation data", required=True, nargs="+")


Anomaly = namedtuple("Anomaly", ["message", "score"])


def prepare_xy_pairs(gen_negatives, data_paths, learning_data):
    xs = []
    ys = []
    # keep calls in addition to encoding as x,y pairs (to report detected anomalies)
    code_pieces = []

    for code_piece in Util.DataReader(data_paths):
        learning_data.code_to_xy_pairs_str(gen_negatives, code_piece, xs, ys,
                                       name_to_vector, "binary", code_pieces)
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


#TransformerBlock
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

if __name__ == '__main__':
    print("BugDetection started with " + str(sys.argv))
    time_start = time.time()
    training_data_paths_list = []
    validation_data_paths_list = []
    args = parser.parse_args()
    name_to_vector_file = args.token_emb
    training_data_paths_list.append(args.training_data_Swapped)
    training_data_paths_list.append(args.training_data_BinOp)
    training_data_paths_list.append(args.training_data_IncBinOp)
    validation_data_paths_list.append(args.validation_data_Swapped)
    validation_data_paths_list.append(args.validation_data_BinOp)
    validation_data_paths_list.append(args.validation_data_IncBinOp)


    with open(name_to_vector_file) as f:
        name_to_vector = json.load(f)

    
    learning_data_objects = []
    learning_data_objects.append(LearningDataSwappedArgs.LearningData())
    learning_data_objects.append(LearningDataBinOperator.LearningData())
    learning_data_objects.append(LearningDataIncorrectBinaryOperand.LearningData())

    all_xs_training = []
    all_ys_training = []
    all_xs_validation = []
    all_ys_validation = []
    all_code_pieces_validation = []
    val_lengths = []

    
    for i in range(len(learning_data_objects)):

        print("Statistics on training data:")
        learning_data_objects[i].pre_scan(training_data_paths_list[i], validation_data_paths_list[i])
        # prepare x,y pairs for learning and validation, therefore generate negatives
        print("Preparing xy pairs for training data:")
        learning_data_objects[i].resetStats()
        xs_training, ys_training, _ = prepare_xy_pairs(
            True, training_data_paths_list[i], learning_data_objects[i])
        
        print("Preparing xy pairs for validation data:")
        learning_data_objects[i].resetStats()
        xs_validation, ys_validation, code_pieces_validation = prepare_xy_pairs(
            True, validation_data_paths_list[i], learning_data_objects[i])
        

        val_lengths.append(len(xs_validation)) # store length of every learning object
        all_xs_training.append(xs_training)  # Append padded tensors to the list
        all_ys_training.append(ys_training)  # Append padded tensors to the list
        all_xs_validation.append(xs_validation)  # Append padded tensors to the list
        all_ys_validation.append(ys_validation)  # Append padded tensors to the list
        all_code_pieces_validation.append(code_pieces_validation)  # Append padded tensors to the list
 
        print("Validation examples   : " + str(len(xs_validation)))
        print("Training examples   : " + str(len(xs_training)))
        print(learning_data_objects[i].stats)
    
    print(val_lengths)
    print(learning_data_objects)

    # Flatten all_xs_training and all_ys_training
    all_xs_training = [item for sublist in all_xs_training for item in sublist]
    all_ys_training = [item for sublist in all_ys_training for item in sublist]
    all_xs_validation = [item for sublist in all_xs_validation for item in sublist]
    all_ys_validation = [item for sublist in all_ys_validation for item in sublist]
    all_code_pieces_validation = [item for sublist in all_code_pieces_validation for item in sublist]

    zip_training = list(zip(all_xs_training, all_ys_training))
    zip_validation = list(zip(all_xs_validation, all_ys_validation, all_code_pieces_validation))
    random.shuffle(zip_training)
    random.shuffle(zip_validation)

    all_xs_training, all_ys_training = zip(*zip_training)
    all_xs_validation, all_ys_validation, all_code_pieces_validation = zip(*zip_validation)

    # all_xs_training to numpy array
    all_xs_training = np.array(all_xs_training)
    all_ys_training = np.array(all_ys_training)
    all_ys_validation = np.array(all_ys_validation)
    all_xs_validation = np.array(all_xs_validation)

    all_xs_training = [PreprocessingData.symbols_to_text(x) for x in all_xs_training]
    all_xs_validation = [PreprocessingData.symbols_to_text(x) for x in all_xs_validation]


    # Tokenize the data training/validation
    tokenizer = Tokenizer(oov_token = 'unknown',filters = '', lower = False)
    tokenizer.fit_on_texts(all_xs_training)
    tokenizer.fit_on_texts(all_xs_validation)
    sequences_train = tokenizer.texts_to_sequences(all_xs_training)
    sequences_val = tokenizer.texts_to_sequences(all_xs_validation)

    # Pad sequences to ensure equal length training
    max_len = max(len(seq) for seq in sequences_train)
    xs_training_padded_sequences = pad_sequences(sequences_train, maxlen=max_len)
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences to ensure equal length validation
    #max_len_val = max(len(seq) for seq in sequences_val)
    xs_validation_padded_sequences = pad_sequences(sequences_val, maxlen=max_len)
    
    print(xs_validation_padded_sequences)



    # create a model (simple feedforward network)
    embed_dim = 128  # Embedding size for each token
    num_heads = 16  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    inputs = Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(200, activation="relu", kernel_initializer='normal')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid", kernel_initializer='normal')(x)

    model = Model(inputs=inputs, outputs=outputs)


    # summarize the model
    print(model.summary())

    #model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(xs_training_padded_sequences, all_ys_training, 
                        batch_size=64, epochs=10, 
                    )

    model.save_weights("predict_class.h5")


    time_stamp = math.floor(time.time() * 1000)
    model.save("bug_detection_model_"+str(time_stamp))

    time_learning_done = time.time()
    print("Time for learning (seconds): " +
          str(round(time_learning_done - time_start)))
    

    #del shuffled_xs_training
    #del shuffled_ys_training
    gc.collect()

     # validate the model
    validation_loss = model.evaluate(xs_validation_padded_sequences, all_ys_validation)
    print()
    print("Validation loss & accuracy: " + str(validation_loss))

 