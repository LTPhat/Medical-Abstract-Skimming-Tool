import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow_hub as hub
from tensorflow.keras.utils import plot_model
import sys
import os


parent_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parent_root)
from config.configs import Params
params = Params()




class AttentionPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionPoolingLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create trainable weights for the attention mechanism
        self.WQ = self.add_weight(name="WQ", shape=(input_shape[-1], input_shape[-1]))
        self.WK = self.add_weight(name="WK", shape=(input_shape[-1], input_shape[-1]))
        self.WV = self.add_weight(name="WV", shape=(input_shape[-1], input_shape[-1]))
        super(AttentionPoolingLayer, self).build(input_shape)


    def call(self, inputs):
        # Compute Q, K, and V vectors from the inputs
        Q = tf.matmul(inputs, self.WQ)
        K = tf.matmul(inputs, self.WK)
        V = tf.matmul(inputs, self.WV)

        # Compute scaled dot-product attention scores
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Compute the weighted sum of V vectors
        output = tf.matmul(attention_weights, V)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1],)
    



class AttentionModel(object):
    def __init__(self, word_vectorizer, word_embed, pretrained_embedding = None, num_classes = 5):
        super(AttentionModel, self).__init__()

        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = num_classes
        self.word_output_dim = params.WORD_OUTPUT_DIM

        # Vectorizer
        self.word_vectorizer = word_vectorizer
        # Embedding
        self.word_embed = word_embed



    def word_level_branch(self, word_input):
        word_vectors = self.word_vectorizer(word_input)
        word_embeddings = self.word_embed(word_vectors)
        x = layers.Dense(128, activation = "relu")(word_embeddings)
        x = layers.BatchNormalization()(x)
        word_outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        return word_outputs


    def fcn(self, total_embed):
          x = layers.Flatten()(total_embed)
          x = layers.Dense(64, activation = "relu")(x)
          x = layers.BatchNormalization()(x)
          x = layers.Dropout(0.5)(x)
          x = layers.Dense(self.num_classes, activation = "softmax")(x)
          return x


    def _get_model(self):

        # Word-token input
        word_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
        word_level_output = self.word_level_branch(word_inputs)

        # Attention layers
        attention_layer = AttentionPoolingLayer()
        output  = attention_layer(word_level_output)

        # Bi-directional decoder
        output = layers.Bidirectional(layers.LSTM(64, return_sequences = True))(output)

        # FCN
        output_layer = self.fcn(output)

        model= tf.keras.Model(inputs=[word_inputs],
                         outputs= output_layer,
                         name="tetra_embeddings_model")
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        return model
    

    def plot_model(self, model):
        plot_model(model)
        return 