import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow_hub as hub
from tensorflow.keras.utils import plot_model
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from config.configs import *
params = Params()


class PentaEmbeddingModel():
    def __init__(self, word_vectorizer = None, char_vectorizer = None, word_embed = None, char_embed = None, pretrained_embedding = None, glove_embed = None, bert_process = None, bert_layer = None, num_classes = 5):
        super(PentaEmbeddingModel, self).__init__()
        """
        word_vectorizer: Word-level vectorizer
        char_vectorizer: Char-level vectorizer
        word_embed: Word-level embedding layer
        char_embed: Char-level embedding layer
        pretrained_embedding: "bert", "glove" or None. Default: None
        glove_embed: glove embedding layer 
        bert_process: BERT input processing layer
        bert_layer: BERT embedding layer
        num_classes: Number of classes. Default: 5 ("Do not change")
        """

        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = params.NUM_CLASSES
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

        # Layers
        self.word_biLSTM = layers.Bidirectional(layers.LSTM(int(self.word_output_dim / 2)))
        self.char_biLSTM = layers.Bidirectional(layers.LSTM(int(self.char_output_dim)))
        self.concat_biLSTM = layers.Bidirectional(layers.LSTM(int(self.concate_dim)))
        self.concatenate = layers.Concatenate()
        self.dense_classes = layers.Dense(self.num_classes, activation = "softmax")
        self.dropout = layers.Dropout(0.5)



#---------- First level branch----------------------
    def word_level_branch(self, word_input):
        """
        Word-token embedding branch
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
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(word_embeddings)
        x = layers.Dense(128, activation = "relu")(x)
        x = layers.BatchNormalization()(x)
        word_outputs = self.word_biLSTM(x)
        return word_outputs


    def char_level_branch(self, char_input):
        """
        arg: 
        - char_input: char-level tokens embedding
        """
        char_vectors = self.char_vectorizer(char_input)
        char_embeddings = self.char_embed(char_vectors)
        x = self.char_biLSTM(char_embeddings)
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

# ----------------Second-level layer----------------------

    def word_char_block(self, word_char_concat):
        """
        Blocks for word-level tokens, char-level tokens
        arg: - word_char_concat: word-level, char-level tokens
        """
        word_char_concat = tf.expand_dims(word_char_concat, axis = 1)
        # LSTM layer for first two concate embeddings
        lstm_concat = self.concat_biLSTM(word_char_concat)
        lstm_concat = layers.Dense(256, activation = "relu")(lstm_concat)
        lstm_concat = layers.Dropout(0.5) (lstm_concat)

        return lstm_concat, lstm_concat.shape

# ---------------Third-level layer-----------------------------

    def sequence_opt_layer(self, total_embed):
        """
        Context-position enrichment layer
        arg:
        - total_embed: 5 concatenated input embedding
        """
        total_embed = tf.expand_dims(total_embed, axis = 1)
        bilstm_out = layers.Bidirectional(layers.LSTM(int(total_embed.shape[-1] / 2)))(total_embed)
        return bilstm_out


    def fcn(self, total_embed):
        """
        Feed forward FCN
        """
        x = layers.Dense(64, activation = "relu", input_shape = (total_embed.shape[1], ))(total_embed)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes, activation = "softmax")(x)
        return x


    def _get_model(self):
        
        # Penta embedding inputs
        word_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
        char_inputs = layers.Input(shape = (1, ), dtype = tf.string, name = "char_input")
        line_ids_inputs = layers.Input(shape = (self.line_ids_input_dim, ), name = "line_ids_input")
        length_lines_inputs = layers.Input(shape = (self.length_lines_input_dim, ), name = "length_lines_input")
        total_lines_inputs = layers.Input(shape = (self.total_lines_input_dim, ), name = "total_lines_input")

        #-----------------------------------------------
        # Branch outputs

        # Word-level
        word_level_output = self.word_level_branch(word_inputs)

        # Char-level branch
        char_level_output = self.char_level_branch(char_inputs)

        #line_ids, length_lines, total_lines branch
        line_ids_output = self.line_ids_branch(line_ids_inputs)
        length_lines_output  = self.length_lines_branch(length_lines_inputs)
        total_lines_output = self.total_lines_branch(total_lines_inputs)
        #---------------------------------------------------------

        #Concate two embeddings
        word_char_concat = self.concatenate([word_level_output,char_level_output])

        # Pass to word_char_block
        word_char_output, word_char_output_shape = self.word_char_block(word_char_concat)
        #--------------------------------------------------------

        # Concanate last three input
        position_embed = self.concatenate([length_lines_output, line_ids_output, total_lines_output])

        # Concatnate 5 input
        total_embed = self.concatenate([word_char_output, position_embed])

        #Sequence label opt layers
        total_embed = self.sequence_opt_layer(total_embed)
        # FCN

        output_layer = self.fcn(total_embed)

        model= tf.keras.Model(inputs=[word_inputs, char_inputs,line_ids_inputs, length_lines_inputs, total_lines_inputs],
                         outputs= output_layer,
                         name="penta_embeddings_model")

        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS, metrics = params.METRICS)
        return model



    def _define_checkpoint(self):
        """
        Define checkpoint for model
        """
        if str(self.pretrained_embedding).lower() == "glove":
            if not os.path.exists(params.PENTA_GLOVE_MODEL_DIR):
                os.makedirs(params.PENTA_GLOVE_MODEL_DIR)
            checkpoint_dir = params.PENTA_GLOVE_MODEL_DIR
        elif str(self.pretrained_embedding).lower() == "bert":
            if not os.path.exists(params.PENTA_BERT_MODEL_DIR):
                os.makedirs(params.PENTA_BERT_MODEL_DIR)
            checkpoint_dir = params.PENTA_BERT_MODEL_DIR
        else:
            if not os.path.exists(params.PENTA_NOR_MODEL_DIR):
                os.makedirs(params.PENTA_NOR_MODEL_DIR)
            checkpoint_dir = params.PENTA_NOR_MODEL_DIR
    
        checkpoint= tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_dir + '/best_model.ckpt',
        monitor = "val_categorical_accuracy",
        save_best_only = True,
        save_weights_only = True,
        verbose = 1
        )
        print("Create checkpoint for penta embeddings model at: ", checkpoint_dir)
        return checkpoint
    

    def _plot_model(self, model):
        plot_model(model)
        return 