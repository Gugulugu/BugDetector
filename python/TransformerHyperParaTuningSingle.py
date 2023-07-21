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
from tensorboard.plugins.hparams import api as hp


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
    xs_training = [PreprocessingData.preprocess_text(x) for x in xs_training]
    #xs_training = [PreprocessingData.symbols_to_text(x) for x in xs_training]
    xs_validation = [PreprocessingData.preprocess_text(x) for x in xs_validation]
    #xs_validation = [PreprocessingData.symbols_to_text(x) for x in xs_validation]
    print(xs_training[0:10])



    tokenizer = Tokenizer(oov_token = 'unknown',filters = '', lower = False)
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


    # Hyperparameter tuning
    # https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

    #HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200, 512]))
    #HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2))
    #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    #HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10, 20, 30, 40, 50]))
    #HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128]))
    #HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.001))
    HP_EMBEDDING_DIM = hp.HParam('embedding_dim', hp.Discrete([4, 256, 512])) #16, 32, 64, 128, 256, 512, 1024 (256 -32 missing)
    HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([2, 16, 32])) #4, 8, 16,
    HP_FF_DIM = hp.HParam('ff_dim', hp.Discrete([4, 256, 512])) #32, 64, 

    METRIC_ACCURACY = 'accuracy'

    

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_EMBEDDING_DIM, HP_NUM_HEADS, HP_FF_DIM],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        
  )
    
    
    def train_test_model(hparams):
        inputs = Input(shape=(max_len_train,))
        embedding_layer = TokenAndPositionEmbedding(max_len_train, vocab_size, hparams[HP_EMBEDDING_DIM])
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(hparams[HP_EMBEDDING_DIM], hparams[HP_NUM_HEADS], hparams[HP_FF_DIM])
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation="relu", kernel_initializer='normal')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation="sigmoid", kernel_initializer='normal')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        #early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto', min_delta=0.01)

        model.fit(xs_training_padded_sequences, ys_training, epochs=10, batch_size = 64) #callbacks=[early_stopping]# Run with 1 epoch to speed things up for demo purposes
        _, accuracy = model.evaluate(xs_validation_padded_sequences, ys_validation)
        return accuracy
    
    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    
    session_num = 0 #110

    #for num_units in HP_NUM_UNITS.domain.values:
        #for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    #for batch_size in HP_BATCH_SIZE.domain.values:
    for embedding_dim in HP_EMBEDDING_DIM.domain.values:
                                for num_heads in HP_NUM_HEADS.domain.values:
                                    for ff_dim in HP_FF_DIM.domain.values:
                                        hparams = {
                                            #HP_NUM_UNITS: num_units,
                                            #HP_DROPOUT: dropout_rate,
                                            #HP_BATCH_SIZE: batch_size,
                                            HP_EMBEDDING_DIM: embedding_dim,
                                            HP_NUM_HEADS: num_heads,
                                            HP_FF_DIM: ff_dim
                                        }
                                        run_name = "run-%d" % session_num
                                        print('--- Starting trial: %s' % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        run('logs/hparam_tuning_swapArgs2/' + run_name, hparams)
                                        session_num += 1
