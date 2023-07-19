import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate, Input, Reshape, BatchNormalization, Activation, Add, Multiply, AveragePooling1D, GlobalAveragePooling1D, LSTM, Bidirectional, TimeDistributed, RepeatVector, Permute, Lambda, Dot, Softmax, Multiply, AdditiveAttention, Attention, Concatenate, Dot, Softmax, Multiply, AdditiveAttention, Attention, Concatenate

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