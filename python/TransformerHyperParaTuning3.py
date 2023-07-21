import sys
import json
from os.path import join
from os import getcwd
from collections import Counter, namedtuple
import math
import argparse
import random

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers.core import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
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
from Transformer import TokenAndPositionEmbedding, TransformerBlock
from Preprocessing import PreprocessingData



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
        learning_data.code_to_xy_pairs_str(gen_negatives, code_piece, xs, ys,
                                       name_to_vector, "multiple", code_pieces)
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
        print(xs_training[0:10])
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


    #preprocess data
    all_xs_training = [PreprocessingData.preprocess_text(x) for x in all_xs_training]
    #all_xs_training = [PreprocessingData.symbols_to_text(x) for x in all_xs_training]
    all_xs_validation = [PreprocessingData.preprocess_text(x) for x in all_xs_validation]
    #all_xs_validation = [PreprocessingData.symbols_to_text(x) for x in all_xs_validation]
    print(all_xs_training[0:10])

    
    # Tokenize the data training/validation
    tokenizer = Tokenizer(oov_token = False,filters = '', lower = True)
    tokenizer.fit_on_texts(all_xs_training)
    tokenizer.fit_on_texts(all_xs_validation)
    sequences_train = tokenizer.texts_to_sequences(all_xs_training)
    sequences_val = tokenizer.texts_to_sequences(all_xs_validation)

    # Pad sequences to ensure equal length training
    max_len = max(len(seq) for seq in sequences_train)
    xs_training_padded_sequences = pad_sequences(sequences_train, maxlen=max_len, padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences to ensure equal length validation
    #max_len_val = max(len(seq) for seq in sequences_val)
    xs_validation_padded_sequences = pad_sequences(sequences_val, maxlen=max_len, padding='post')

        
    print(xs_training_padded_sequences[0:10])
    classes = len(np.unique(all_ys_training))




    # Hyperparameter tuning
    # https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

    #HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200, 512]))
    #HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2))
    #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    #HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10, 20, 30, 40, 50]))
    #HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128]))
    #HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.001))
    HP_EMBEDDING_DIM = hp.HParam('embedding_dim', hp.Discrete([32, 256, 512])) #16, 32, 64, 128, 256, 512, 1024 (256 -32 missing)
    HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([8, 16, 32])) #4, 8, 16,
    HP_FF_DIM = hp.HParam('ff_dim', hp.Discrete([64, 256, 512])) #32, 64, 

    METRIC_ACCURACY = 'accuracy'

    

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_EMBEDDING_DIM, HP_NUM_HEADS, HP_FF_DIM],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        
  )
    
    
    def train_test_model(hparams):
        inputs = Input(shape=(max_len,))
        embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, hparams[HP_EMBEDDING_DIM])
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(hparams[HP_EMBEDDING_DIM], hparams[HP_NUM_HEADS], hparams[HP_FF_DIM])
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation="relu", kernel_initializer='normal')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(classes, activation="softmax", kernel_initializer='normal')(x)

        model = Model(inputs=inputs, outputs=outputs)

        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=6000, decay_rate=0.96, staircase=True
        )

        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )



        #early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto', min_delta=0.01)

        model.fit(xs_training_padded_sequences, all_ys_training, epochs=10, batch_size = 64) #callbacks=[early_stopping]# Run with 1 epoch to speed things up for demo purposes
        _, accuracy = model.evaluate(xs_validation_padded_sequences, all_ys_validation)
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
                                        run('logs/hparam_tuning_old_approach/' + run_name, hparams)
                                        session_num += 1
