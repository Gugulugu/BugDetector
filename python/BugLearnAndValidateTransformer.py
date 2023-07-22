import sys
import json
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math
import argparse

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
import LearningDataIncorrectArgsType
from Transformer import TransformerBlock, TokenAndPositionEmbedding
from Preprocessing import PreprocessingData


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pattern", help="Kind of data to extract", choices=["SwappedArgs", "BinOperator", "SwappedBinOperands", "IncorrectBinaryOperand", "IncorrectAssignment", "IncorrectArgsType"], required=True)
parser.add_argument(
    "--token_emb", help="JSON file with token embeddings", required=True)
parser.add_argument(
    "--training_data", help="JSON files with training data", required=True, nargs="+")
parser.add_argument(
    "--validation_data", help="JSON files with validation data", required=True, nargs="+")


Anomaly = namedtuple("Anomaly", ["message", "score"])


def prepare_xy_pairs(gen_negatives, data_paths, learning_data):
    xs = []
    ys = []
    # keep calls in addition to encoding as x,y pairs (to report detected anomalies)
    code_pieces = []

    for code_piece in Util.DataReader(data_paths):
        learning_data.code_to_xy_pairs_str(gen_negatives, code_piece, xs, ys,
                                       name_to_vector,"binary", code_pieces)
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

    args = parser.parse_args()
    pattern = args.pattern
    name_to_vector_file = args.token_emb
    training_data_paths = args.training_data
    validation_data_paths = args.validation_data

    with open(name_to_vector_file) as f:
        name_to_vector = json.load(f)

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
    elif pattern == "IncorrectArgsType":
        learning_data = LearningDataIncorrectArgsType.LearningData()
    else:
        raise Exception(f"Unexpected bug pattern: {pattern}")
    # not yet implemented
    # elif pattern == "MissingArg":
    ##    learning_data = LearningDataMissingArg.LearningData()

    print("Statistics on training data:")
    learning_data.pre_scan(training_data_paths, validation_data_paths)

    # prepare x,y pairs for learning and validation, therefore generate negatives
    print("Preparing xy pairs for training data:")
    learning_data.resetStats()
    xs_training, ys_training, _ = prepare_xy_pairs(
        True, training_data_paths, learning_data)
    x_length = len(xs_training[0])
    print("Training examples   : " + str(len(xs_training)))
    print(learning_data.stats)

    # prepare validation data
    print("Preparing xy pairs for validation data:")
    learning_data.resetStats()
    xs_validation, ys_validation, code_pieces_validation = prepare_xy_pairs(
        True, validation_data_paths, learning_data)
    print("Validation examples : " + str(len(xs_validation)))
    print(learning_data.stats)

    print(xs_training[0:10])

    #preprocess data
    #xs_training = [PreprocessingData.preprocess_text(x) for x in xs_training]
    #xs_training = [PreprocessingData.symbols_to_text(x) for x in xs_training]
    #xs_validation = [PreprocessingData.preprocess_text(x) for x in xs_validation]
    #xs_validation = [PreprocessingData.symbols_to_text(x) for x in xs_validation]
    print(xs_training[0:10])



    tokenizer = Tokenizer(oov_token = 'unknown',filters = ':', lower = True)
    tokenizer.fit_on_texts(xs_training)
    tokenizer.fit_on_texts(xs_validation)
    sequences_train = tokenizer.texts_to_sequences(xs_training)
    sequences_val = tokenizer.texts_to_sequences(xs_validation)

    # Pad sequences to ensure equal length training
    max_len_train = max(len(seq) for seq in sequences_train)
    xs_training_padded_sequences = pad_sequences(sequences_train, maxlen=max_len_train, padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences to ensure equal length validation
    #max_len_val = max(len(seq) for seq in sequences_val)
    xs_validation_padded_sequences = pad_sequences(sequences_val, maxlen=max_len_train, padding='post')
    print(xs_training_padded_sequences[0:10])
    print(xs_validation_padded_sequences.shape)

    # create a model (simple feedforward network)
    embed_dim = 64  # Embedding size for each token
    num_heads = 8  # Number of attention heads
    ff_dim = 16  # Hidden layer size in feed forward network inside transformer

    inputs = Input(shape=(max_len_train,))
    embedding_layer = TokenAndPositionEmbedding(max_len_train, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(200, activation="relu", kernel_initializer="normal")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid",kernel_initializer="normal")(x)

    model = Model(inputs=inputs, outputs=outputs)


    # summarize the model
    print(model.summary())

    initial_learning_rate = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1600, decay_rate=0.96, staircase=True
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


    #model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy', 'Precision', 'Recall'])
    
    history = model.fit(xs_training_padded_sequences, ys_training, 
                        batch_size=64, epochs=10, validation_data = (xs_validation_padded_sequences, ys_validation) 
                    )

    model.save_weights("predict_class.h5")


    time_stamp = math.floor(time.time() * 1000)
    model.save("bug_detection_model_"+str(time_stamp))

    time_learning_done = time.time()
    print("Time for learning (seconds): " +
          str(round(time_learning_done - time_start)))


    # validate the model
    validation_loss = model.evaluate(xs_validation_padded_sequences, ys_validation)
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
    ys_prediction = model.predict(xs_validation_padded_sequences)
    poss_anomalies = []
    for idx in range(0, len(xs_validation_padded_sequences), 2):
        # probab(original code should be changed), expect 0
        y_prediction_orig = ys_prediction[idx][0]
        # probab(changed code should be changed), expect 1
        y_prediction_changed = ys_prediction[idx + 1][0]
        # higher means more likely to be anomaly in current code
        anomaly_score = learning_data.anomaly_score(
            y_prediction_orig, y_prediction_changed)
        # higher means more likely to be correct in current code
        normal_score = learning_data.normal_score(
            y_prediction_orig, y_prediction_changed)
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
            code_piece = code_pieces_validation[idx]
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
            threshold_to_found_seeded_bugs[threshold] * 1.0) / (len(xs_validation_padded_sequences) / 2)
        precision = 1 - \
            ((threshold_to_warnings_in_orig_code[threshold]
              * 1.0) / (len(xs_validation_padded_sequences) / 2))
        if threshold_to_correct[threshold] + threshold_to_incorrect[threshold] > 0:
            accuracy = threshold_to_correct[threshold] * 1.0 / (
                threshold_to_correct[threshold] + threshold_to_incorrect[threshold])
        else:
            accuracy = 0.0
        print("Threshold: " + str(threshold) + "   Accuracy: " + str(round(accuracy, 4)) + "   Recall: " + str(round(recall, 4)
                                                                                                               ) + "   Precision: " + str(round(precision, 4))+"  #Warnings: "+str(threshold_to_warnings_in_orig_code[threshold]))
