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

## ATENTION-BASED WORD_INPUT MODEL


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
    def __init__(self, word_vectorizer, word_embed, pretrained_embedding = None, glove_embed = None, bert_process = None, bert_layer = None, num_classes = 5):
        super(AttentionModel, self).__init__()

        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = num_classes
        self.word_output_dim = params.WORD_OUTPUT_DIM

        # Vectorizer
        self.word_vectorizer = word_vectorizer
        # Embedding
        self.word_embed = word_embed
        self.glove_embed = glove_embed
        self.bert_process =bert_process
        self.bert_layer = bert_layer


    def word_level_branch(self, word_input):
        """
        Word-token embedding branch
        """
        if str(self.pretrained_embedding).lower() == "bert":
            # Pretrained Bert embeddings
            bert_input = self.bert_process(word_input)
            bert_output = self.bert_layer(bert_input, training = False)
            word_embeddings = bert_output['sequence_output']
        else:
            if (self.word_vectorizer):    
                if (str(self.pretrained_embedding).lower() == "glove"):
                    # Get glove embedding
                    word_vectors = self.word_vectorizer(word_input)
                    word_embeddings = self.glove_embed(word_vectors)
                else:
                    # Original word_embeddings
                    word_vectors = self.word_vectorizer(word_input)
                    word_embeddings = self.word_embed(word_vectors)
            else:
                raise Exception("Please provide word vectorizer.")
    
        x = layers.Dense(128, activation = "relu")(word_embeddings)
        x = layers.BatchNormalization()(x)
        word_outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        return word_outputs
    
        # word_vectors = self.word_vectorizer(word_input)
        # word_embeddings = self.word_embed(word_vectors)
        # x = layers.Dense(128, activation = "relu")(word_embeddings)
        # x = layers.BatchNormalization()(x)
        # word_outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        # return word_outputs
    


    def fcn(self, total_embed):
        """
        Fully-connected block
        """
        x = layers.Flatten()(total_embed)
        x = layers.Dense(64, activation = "relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation = "softmax")(x)
        return output


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
                         name="attention_base_model")
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        return model
    

    def _define_checkpoint(self):
        """
        Define checkpoint
        """
        if str(self.pretrained_embedding).lower() == "glove":
            if not os.path.exists(params.WORD_MODEL_ATT_GLOVE_DIR):
                os.makedirs(params.WORD_MODEL_ATT_GLOVE_DIR)
            checkpoint_dir = params.WORD_MODEL_ATT_GLOVE_DIR
        elif str(self.pretrained_embedding).lower() == "bert":
            if not os.path.exists(params.WORD_MODEL_ATT_BERT_DIR):
                os.makedirs(params.WORD_MODEL_ATT_BERT_DIR)
            checkpoint_dir = params.WORD_MODEL_ATT_BERT_DIR
        else:
            if not os.path.exists(params.WORD_MODEL_ATT_NOR_DIR):
                os.makedirs(params.WORD_MODEL_ATT_NOR_DIR)
            checkpoint_dir = params.WORD_MODEL_ATT_NOR_DIR

        checkpoint= tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_dir + '/best_model.ckpt',
        monitor = "val_categorical_accuracy",
        save_best_only = True,
        save_weights_only = True,
        verbose = 1
        )
        print("Create checkpoint for Attention-based model at: ", checkpoint_dir)
        
        return checkpoint
    
    
    def plot_model(self, model):
        plot_model(model)
        return 