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




class HybridEmbeddingModel(object):
    def __init__(self, word_vectorizer = None, char_vectorizer = None, word_embed = None, char_embed = None, 
                 pretrained_embedding = "None", glove_embed = None, bert_process = None, bert_layer = None, num_classes = 5):
        """
        word_vectorizer: Word-level vectorizer
        char_vectorizer: Char-level vectorizer
        word_embed: Word-level embedding layer
        char_embed: Char-level embedding layer
        pretrained_embedding: "bert", "glove" or None. Default: None
        num_classes: Number of classes. Default: 5 ("Do not change")
        """
        super(HybridEmbeddingModel, self).__init__()
        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = params.NUM_CLASSES
        self.word_output_dim = params.WORD_OUTPUT_DIM
        self.char_output_dim = params.CHAR_OUTPUT_DIM
        self.concate_dim = self.word_output_dim + 2 * self.char_output_dim

        # Vectorizer
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer

        # Embedding
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.glove_embed = glove_embed
        self.bert_process = bert_process
        self.bert_layer = bert_layer

        # Layers
        self.word_biLSTM = layers.Bidirectional(layers.LSTM(int(self.word_output_dim / 2)))
        self.char_biLSTM = layers.Bidirectional(layers.LSTM(int(self.char_output_dim)))
        self.concat_biLSTM = layers.Bidirectional(layers.LSTM(int(self.concate_dim / 2)))
        self.concatenate = layers.Concatenate()
        self.batchnorm = layers.BatchNormalization()
        self.fcn = Sequential([
            layers.Input(shape = (self.concate_dim, )),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation = "softmax")
        ])
        # Model
        self.model = None



    def word_level_branch(self, word_input):

        if str(self.pretrained_embedding).lower() == "bert":
            # Pretrained BERT embeddings
            bert_input = self.bert_process(word_input)
            bert_output = self.bert_layer(bert_input, training = False)
            word_embeddings = bert_output['sequence_output']
        else:
            word_vectors = self.word_vectorizer(word_input)
            if str(self.pretrained_embedding).lower() == "glove":
                # Pretrained glove embeddings
                word_embeddings = self.glove_embed(word_vectors)
            else:
                # Original word_embeddings
                word_embeddings = self.word_embed(word_vectors)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(word_embeddings)
        x = layers.Dense(128, activation = "relu")(x)
        x = layers.BatchNormalization()(x)
        word_outputs = self.word_biLSTM(x)
        return word_outputs
    


    def char_level_branch(self, char_input):

        char_vectors = self.char_vectorizer(char_input)
        char_embeddings = self.char_embed(char_vectors)
        x = self.char_biLSTM(char_embeddings)
        return x


    def _get_model(self):

        # Input
        word_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
        char_inputs = layers.Input(shape = (1, ), dtype = tf.string, name = "char_input")


        # Word-level branch
        word_level_output = self.word_level_branch(word_inputs)
        # Char-level branch
        char_level_output = self.char_level_branch(char_inputs)


        # Concate two embeddings
        word_char_concat = self.concatenate([word_level_output,char_level_output])

        word_char_concat = tf.expand_dims(word_char_concat, axis = 1)


        # LSTM layer for concate embeddings
        lstm_concat = self.concat_biLSTM(word_char_concat)

        # FCN
        output_layer = self.fcn(lstm_concat)

        model= tf.keras.Model(inputs=[word_inputs, char_inputs],
                         outputs= output_layer,
                         name="hybrid_embeddings_model")
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        return model
    

    def _define_checkpoint(self):
        pass


    def plot_model(self, model):
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        plot_model(model)
        return 
    

    @staticmethod
    def _fit_model(model):
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        print(model.summary())
        model_history = model.fit()
        return model_history
    
