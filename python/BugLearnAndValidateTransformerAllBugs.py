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
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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



import time
import numpy as np
import Util
import LearningDataSwappedArgs
import LearningDataBinOperator
import LearningDataSwappedBinOperands
import LearningDataIncorrectBinaryOperand
import LearningDataIncorrectAssignment
import TransformerBlock
import TokenAndPositionEmbedding





parser = argparse.ArgumentParser()
parser.add_argument(
    "--token_emb", help="JSON file with token embeddings", required=True)
parser.add_argument(
    "--type_emb", help="JSON file with type embeddings", required=True)
parser.add_argument(
    "--node_emb", help="JSON file with AST node embeddings", required=True)
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
    print("BugDetection started with " + str(sys.argv))
    time_start = time.time()
    training_data_paths_list = []
    validation_data_paths_list = []
    args = parser.parse_args()
    name_to_vector_file = args.token_emb
    type_to_vector_file = args.type_emb
    node_type_to_vector_file = args.node_emb
    training_data_paths_list.append(args.training_data_Swapped)
    training_data_paths_list.append(args.training_data_BinOp)
    training_data_paths_list.append(args.training_data_IncBinOp)
    validation_data_paths_list.append(args.validation_data_Swapped)
    validation_data_paths_list.append(args.validation_data_BinOp)
    validation_data_paths_list.append(args.validation_data_IncBinOp)


    with open(name_to_vector_file) as f:
        name_to_vector = json.load(f)
    with open(type_to_vector_file) as f:
        type_to_vector = json.load(f)
    with open(node_type_to_vector_file) as f:
        node_type_to_vector = json.load(f)

    
    learning_data_objects = []
    learning_data_objects.append(LearningDataSwappedArgs.LearningData())
    learning_data_objects.append(LearningDataBinOperator.LearningData())
    learning_data_objects.append(LearningDataIncorrectBinaryOperand.LearningData())

    all_xs_training = []
    all_ys_training = []
    
    for i in range(len(learning_data_objects)):

        print("Statistics on training data:")
        learning_data_objects[i].pre_scan(training_data_paths_list[i], validation_data_paths_list[i])
        # prepare x,y pairs for learning and validation, therefore generate negatives
        print("Preparing xy pairs for training data:")
        learning_data_objects[i].resetStats()
        xs_training, ys_training, _ = prepare_xy_pairs(
            True, training_data_paths_list[i], learning_data_objects[i])
        

        xs_training_padded = tf.pad(xs_training, [[0, 0], [0, 1210 - xs_training.shape[1]]], constant_values=0)  # Pad xs_training with zeros to match [x, 1210] shape

        all_xs_training.append(xs_training_padded)  # Append padded tensors to the list
        all_ys_training.append(ys_training)  # Append padded tensors to the list

        print("Training examples   : " + str(len(xs_training)))
        print(learning_data_objects[i].stats)

    # combine all training data to one tensor
    combined_xs_training = tf.concat(all_xs_training, axis=0)
    combined_ys_training = tf.concat(all_ys_training, axis=0)
    print(tf.shape(combined_xs_training))
    print('shape of array :', combined_xs_training.shape)


    x_length = len(combined_xs_training[0]) 


    # create a model (simple feedforward network)
    embed_dim = 64  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    inputs = Input(shape=(x_length,))
    embedding_layer = TokenAndPositionEmbedding(x_length, 10000, embed_dim)
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
    
    history = model.fit(combined_xs_training, combined_ys_training, 
                        batch_size=32, epochs=10, 
                    )

    model.save_weights("predict_class.h5")


    time_stamp = math.floor(time.time() * 1000)
    model.save("bug_detection_model_"+str(time_stamp))

    time_learning_done = time.time()
    print("Time for learning (seconds): " +
          str(round(time_learning_done - time_start)))

    # prepare validation data
    all_xs_validation = []
    all_ys_validation = []
    all_code_pieces_validation = []
    
    for i in range(len(learning_data_objects)):
        learning_data_objects[i].resetStats()
      # prepare x,y pairs for learning and validation, therefore generate negatives
        print("Preparing xy pairs for training data:")
        learning_data_objects[i].resetStats()
        xs_validation, ys_validation, code_pieces_validation = prepare_xy_pairs(
            True, validation_data_paths_list[i], learning_data_objects[i])
        

        xs_validation_padded = tf.pad(xs_validation, [[0, 0], [0, 1210 - xs_validation.shape[1]]], constant_values=0)  # Pad xs_training with zeros to match [x, 1210] shape

        all_xs_validation.append(xs_validation_padded)  # Append padded tensors to the list
        all_ys_validation.append(ys_validation)  # Append padded tensors to the list
        all_code_pieces_validation.append(code_pieces_validation)  # Append padded tensors to the list
        
        print("Validation examples   : " + str(len(xs_validation)))
        print(learning_data_objects[i].stats)

    # combine all validation data to one tensor
    combined_xs_validation = tf.concat(all_xs_validation, axis=0)
    combined_ys_validation = tf.concat(all_ys_validation, axis=0)
    # flatten all code pieces
    all_code_pieces_validation = [item for sublist in all_code_pieces_validation for item in sublist]

    # validate the model
    validation_loss = model.evaluate(combined_xs_validation, combined_ys_validation)
    print()
    print("Validation loss & accuracy: " + str(validation_loss))

    # compute precision and recall with different thresholds
    #  for reporting anomalies
    # assumption: correct and incorrect arguments are alternating
    #  in list of x-y pairs
    threshold_to_correct = Counter()
    threshold_to_incorrect = Counter()
    threshold_to_found_seeded_bugs = Counter()
    threshold_to_warnings_in_orig_code = Counter()
    ys_prediction = model.predict(combined_xs_validation)
    poss_anomalies = []
    for idx in range(0, len(combined_xs_validation), 2):
        # probab(original code should be changed), expect 0
        y_prediction_orig = ys_prediction[idx][0]
        # probab(changed code should be changed), expect 1
        y_prediction_changed = ys_prediction[idx + 1][0]
        # higher means more likely to be anomaly in current code
        anomaly_score = y_prediction_orig
        # higher means more likely to be correct in current code
        normal_score = y_prediction_changed
        is_anomaly = False
        for threshold_raw in range(1, 20, 1):
            threshold = threshold_raw / 20.0
            suggests_change_of_orig = anomaly_score >= threshold
            suggests_change_of_changed = normal_score >= threshold
            # counts for positive example
            if suggests_change_of_orig:
                threshold_to_incorrect[threshold] += 1
                threshold_to_warnings_in_orig_code[threshold] += 1
            else:
                threshold_to_correct[threshold] += 1
            # counts for negative example
            if suggests_change_of_changed:
                threshold_to_correct[threshold] += 1
                threshold_to_found_seeded_bugs[threshold] += 1
            else:
                threshold_to_incorrect[threshold] += 1

            # check if we found an anomaly in the original code
            if suggests_change_of_orig:
                is_anomaly = True

        if is_anomaly:
            code_piece = all_code_pieces_validation[idx]
            message = "Score : " + \
                str(anomaly_score) + " | " + code_piece.to_message()
#             print("Possible anomaly: "+message)
            # Log the possible anomaly for future manual inspection
            poss_anomalies.append(Anomaly(message, anomaly_score))

    f_inspect = open('poss_anomalies.txt', 'w+')
    poss_anomalies = sorted(poss_anomalies, key=lambda a: -a.score)
    for anomaly in poss_anomalies:
        f_inspect.write(anomaly.message + "\n")
    print("Possible Anomalies written to file : poss_anomalies.txt")
    f_inspect.close()

    time_prediction_done = time.time()
    print("Time for prediction (seconds): " +
          str(round(time_prediction_done - time_learning_done)))

    print()
    for threshold_raw in range(1, 20, 1):
        threshold = threshold_raw / 20.0
        recall = (
            threshold_to_found_seeded_bugs[threshold] * 1.0) / (len(xs_validation) / 2)
        precision = 1 - \
            ((threshold_to_warnings_in_orig_code[threshold]
              * 1.0) / (len(xs_validation) / 2))
        if threshold_to_correct[threshold] + threshold_to_incorrect[threshold] > 0:
            accuracy = threshold_to_correct[threshold] * 1.0 / (
                threshold_to_correct[threshold] + threshold_to_incorrect[threshold])
        else:
            accuracy = 0.0
        print("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + "   Recall: " + str(round(recall, 4)
                                                                                                               ) + "   Precision: " + str(round(precision, 4))+"  #Warnings: "+str(threshold_to_warnings_in_orig_code[threshold]))
