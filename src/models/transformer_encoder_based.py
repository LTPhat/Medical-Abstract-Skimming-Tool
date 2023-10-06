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


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout_rate=0.1):
        """
        args:
        - d_model: Embedding dim
        - nhead: Number of heads in MultiHeadAttention
        - dim_feedforward: Dense layer's dim
        """
        super(TransformerEncoderLayer, self).__init__()

        self.attention = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model // nhead)
        self.feed_forward = tf.keras.Sequential([
            layers.Dense(dim_feedforward, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)



    def call(self, inputs, training=True, mask=None):
        attention_output = self.attention(inputs, inputs, inputs, attention_mask=mask)
        attention_output = self.dropout(attention_output)
        output1 = self.norm1(inputs + attention_output)

        feed_forward_output = self.feed_forward(output1)
        feed_forward_output = self.dropout(feed_forward_output)
        output2 = self.norm2(output1 + feed_forward_output)

        return output2



class TransformerEncoder(tf.keras.layers.Layer):
    """
    Stack of TransformerEncoderLayer
    """

    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_rate)
                               for _ in range(num_layers)]


    def call(self, inputs, training=True, mask=None):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training, mask=mask)
        return x



class TransformerModel(object): 
    def __init__(self, word_vectorizer = None, char_vectorizer = None, word_embed = None, char_embed= None, 
                 num_layers = None, d_model = None, nhead = None, dim_feedforward = None, pretrained_embedding = None, 
                 glove_embed = None, bert_process = None, bert_layer = None, dropout_rate=0.1, num_classes=5):
        super(TransformerModel, self).__init__()
        """
        args:
        - word_vectorizer: Word-level vectorizer
        - char_vectorizer: Char-level vectorizer
        - word_embed: Word-level embedding layer
        - char_embed: Char-level embedding layer
        - pretrained_embedding: "bert", "glove" or None. Default: None
        - glove_embed: glove embedding layer 
        - bert_process: BERT input processing layer
        - bert_layer: BERT embedding layer
        - num_classes: Number of classes. Default: 5 ("Do not change")

        - num_layers: Number of TransformerEncoder in the model
        - d_model: Embedding dim
        - nhead: Number of heads in MultiHeadAttention
        - dim_feedforward: Dense layer's dim
        """
        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = num_classes
        self.word_output_dim = params.WORD_OUTPUT_DIM
        self.char_output_dim = params.CHAR_OUTPUT_DIM
        self.concate_dim = self.word_output_dim + 2 * self.char_output_dim

        self.line_ids_input_dim = params.LINE_IDS_DEPTH
        self.length_lines_input_dim = params.LENGTH_LINES_DEPTH
        self.total_lines_input_dim = params.TOTAL_LINES_DEPTH


        # Vectorizer
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer


        # Embedding
        self.word_embed = word_embed
        self.char_embed = char_embed
        self.glove_embed = glove_embed
        self.bert_process = bert_process
        self.bert_layer = bert_layer

        # Define the TransformerEncoder
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout_rate)



    def word_level_branch(self, word_input):
        """
        Pretrained BERT don't need vectorization layer
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
        word_outputs = layers.BatchNormalization()(x)

        return word_outputs


    def char_level_branch(self, char_input):
        """
        arg: 
        - char_input: char-level tokens embedding
        """
        char_vectors = self.char_vectorizer(char_input)
        char_embeddings = self.char_embed(char_vectors)
        x = layers.Dense(128, activation ="relu")(char_embeddings)
        x = layers.BatchNormalization()(x)
        return x



    def line_ids_branch(self, line_id_in):
        """
        arg: 
        - line_id_in: line_ids one-hot embedding
        """

        x = layers.Dense(64, activation = "relu")(line_id_in)
        x = layers.BatchNormalization()(x)
        return x



    def length_lines_branch(self, length_line_in):
        """
        arg:
        - length_line_in: length_lines one-hot embedding
        """

        x = layers.Dense(64, activation = "relu")(length_line_in)
        x = layers.BatchNormalization()(x)
        return x


    def total_lines_branch(self, total_line_in):
        """
        arg:
        -total_lines_in: total_lines one-hot embedding
        """

        x = layers.Dense(64, activation = "relu")(total_line_in)
        x = layers.BatchNormalization()(x)
        return x


    def _get_model(self):

        # 5 inputs
            ## Token-inputs
        word_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
        char_inputs = layers.Input(shape = (1, ), dtype = tf.string, name = "char_input")
            ## Positional inputs
        line_ids_inputs = layers.Input(shape = (self.line_ids_input_dim, ), name = "line_ids_input")
        length_lines_inputs = layers.Input(shape = (self.length_lines_input_dim, ), name = "length_lines_input")
        total_lines_inputs = layers.Input(shape = (self.total_lines_input_dim, ), name = "total_lines_input")

        #-----------------------------------------------
        # Branch outputs
            ## Word-level
        word_level_output = self.word_level_branch(word_inputs)

            ## Char-level branch
        char_level_output = self.char_level_branch(char_inputs)

            ##line_ids, length_lines, total_lines branch
        line_ids_output = self.line_ids_branch(line_ids_inputs)
        length_lines_output  = self.length_lines_branch(length_lines_inputs)
        total_lines_output = self.total_lines_branch(total_lines_inputs)
        #---------------------------------------------------------

        #Concate two tokens-embeddings
        word_char_concat = tf.concat([word_level_output,char_level_output], axis = 1)

        #----------------------------------------------------------
        # Concanate last three input
        position_embed = tf.concat([length_lines_output, line_ids_output, total_lines_output], axis = 1)

        # Reshape axis = 2 dimension
        position_embed = layers.Dense(128, activation = "relu")(position_embed)

        # Expand-dim
        position_embed = tf.expand_dims(position_embed, axis = 1)
        #------------------------------------------------------------
        
        # concatenate 5 inputs
        total_embed = tf.concat([word_char_concat, position_embed], axis = 1)

        # TransformerEncoder layer
        x = self.encoder(total_embed, training=True, mask= None)

        # Bi-LSTM decoder layer
        x = layers.Bidirectional(layers.LSTM(64))(x)
        

        # FCN
        x = layers.Dense(64, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output=  layers.Dense(self.num_classes, activation = "softmax")(x)
        model= tf.keras.Model(inputs=[word_inputs, char_inputs, line_ids_inputs, length_lines_inputs, total_lines_inputs],
                        outputs= output,
                        name="transformer_encoder_based_model")

        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS, metrics = params.METRICS)
        
        return model

    def _define_checkpoint(self):
        """
        Define checkpoint for model
        """

        if str(self.pretrained_embedding).lower() == "glove":
            if not os.path.exists(params.TF_BASED_GLOVE_MODEL_DIR):
                os.makedirs(params.TF_BASED_GLOVE_MODEL_DIR)
            checkpoint_dir = params.TF_BASED_GLOVE_MODEL_DIR
        elif str(self.pretrained_embedding).lower() == "bert":
            if not os.path.exists(params.TF_BASED_BERT_MODEL_DIR):
                os.makedirs(params.TF_BASED_BERT_MODEL_DIR)
            checkpoint_dir = params.TF_BASED_BERT_MODEL_DIR
        else:
            if not os.path.exists(params.TF_BASED_NOR_MODEL_DIR):
                os.makedirs(params.TF_BASED_NOR_MODEL_DIR)
            checkpoint_dir = params.TF_BASED_NOR_MODEL_DIR
    
        checkpoint= tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_dir + '/best_model.ckpt',
        monitor = "val_categorical_accuracy",
        save_best_only = True,
        save_weights_only = True,
        verbose = 1
        )
        return checkpoint


    def _plot_model(self, model):
        plot_model(model)
        return 
