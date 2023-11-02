import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.utils import plot_model
import sys
import os


# ---------------------Hierarchy BiLSTM -----------------------------


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from config.configs import *
params = Params()



class MultipleConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim = 128, num_filters = 256, filter_sizes = [2, 3, 5], dropout_rate = 0.5):
        super(MultipleConv1DBlock, self).__init__()
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate
        self.dense = tf.keras.layers.Dense(output_dim, activation = "relu")
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.conv_layers = []

        for filter_size in self.filter_sizes:
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=self.num_filters,
                                               kernel_size=filter_size,
                                               activation='relu'))


    def call(self, inputs):
        conv_outputs = []
        for layer in self.conv_layers:
            conv_layer = layer(inputs)
            conv_outputs.append(conv_layer)

        if len(self.filter_sizes) > 1:
            concat_layer = tf.concat(conv_outputs, axis = 1)
        else:
            concat_layer = conv_outputs[0]

        dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)(concat_layer)
        output = self.dense(dropout_layer)
        output = self.batchnorm(output)
        return output
    



class BiLSTMFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self,output_dim = 128 ,num_units = 128, num_layers = 3, return_sequences=False, dropout=0.5):
        super(BiLSTMFeatureExtractor, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.bi_lstms = [tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.num_units, return_sequences=True, dropout=self.dropout),
            merge_mode='concat'
        ) for _ in range(self.num_layers - 1)]

        self.last_bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.num_units, return_sequences=False, dropout=self.dropout),
            merge_mode='concat')

        self.dense1 = tf.keras.layers.Dense(2 * output_dim, activation = "relu")
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(output_dim, activation = "relu")
        self.batchnorm2 = tf.keras.layers.BatchNormalization()


    def call(self, inputs):
        features = inputs
        for bi_lstm in self.bi_lstms:
            features = bi_lstm(features)

        output = self.last_bi_lstm(features)
        output = self.dense1(output)
        output = self.batchnorm1(output)
        output = self.dense2(output)
        output = self.batchnorm2(output)

        return output



class MLPSharedBlock(tf.keras.layers.Layer):
      def __init__(self, output_dim = 128 , dims = [256, 256, 128], return_sequences=False, dropout=0.5):
          super(MLPSharedBlock, self).__init__()
          self.dense1 = tf.keras.layers.Dense(dims[0], activation = "relu")
          self.batchnorm1 = tf.keras.layers.BatchNormalization()
          self.dense2 = tf.keras.layers.Dense(dims[1], activation = "relu")
          self.batchnorm2 = tf.keras.layers.BatchNormalization()
          self.dense3 = tf.keras.layers.Dense(dims[2], activation = "relu")
          self.batchnorm3 = tf.keras.layers.BatchNormalization()



      def call(self, inputs):
          output = self.dense1(inputs)
          output = self.batchnorm1(output)
          output = layers.Dropout(0.5)(output)
          output = self.dense2(output)
          output = self.batchnorm2(output)
          output = layers.Dropout(0.5)(output)
          output = self.dense3(output)
          output = self.batchnorm3(output)
          return output


class HierarchyBiLSTM():
    def __init__(self, word_vectorizer, char_vectorizer, word_embed, char_embed, pretrained_embedding = None,
                 glove_embed = None, bert_process = None, bert_layer = None, num_classes = 5):
        super(HierarchyBiLSTM, self).__init__()

        # Params
        self.pretrained_embedding = pretrained_embedding
        self.num_classes = params.NUM_CLASSES
        self.word_output_dim = params.WORD_OUTPUT_DIM
        self.char_output_dim = params.CHAR_OUTPUT_DIM

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

        self.concat_biLSTM = layers.Bidirectional(layers.LSTM(128, return_sequences = False))
        self.concatenate = layers.Concatenate()
        self.dense_classes = layers.Dense(self.num_classes, activation = "softmax")
        self.dropout = layers.Dropout(0.5)

        # self.conv1dBlock = MultipleConv1DBlock()

        self.char_extractor = BiLSTMFeatureExtractor()

        self.word_extractor = BiLSTMFeatureExtractor()

        self.concat_extractor = BiLSTMFeatureExtractor()

        self.mlp_share1 = MLPSharedBlock()



#---------- First level branch----------------------
    def word_level_branch(self, word_input):
        if str(self.pretrained_embedding).lower() == "bert":
            # Pretrained Bert embeddings
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
        # Experiment with Conv1d or not
        # word_outputs = self.conv1dBlock(word_embeddings)
        word_outputs = self.word_extractor(word_embeddings)
        return word_outputs
    
    
    def char_level_branch(self, char_input):

        char_vectors = self.char_vectorizer(char_input)
        char_embeddings = self.char_embed(char_vectors)
        x = self.char_extractor(char_embeddings)
        return x

# ----------------Second-level layer----------------------

    def word_char_block(self, word_char_concat):

        word_char_concat = tf.expand_dims(word_char_concat, axis = 1)

        # lstm_concat = self.concat_biLSTM(word_char_concat)
        lstm_concat = self.concat_extractor(word_char_concat)

        return lstm_concat


# ---------------Third-level layer-----------------------------

    def sequence_opt_layer(self, total_embed):
        total_embed = tf.expand_dims(total_embed, axis = 1)
        bilstm_out = layers.Bidirectional(layers.LSTM(int(total_embed.shape[-1] / 2)))(total_embed)
        return bilstm_out


    def fcn(self, total_embed):
          x = layers.Dense(128, activation = "relu", input_shape = (total_embed.shape[1], ))(total_embed)
          x = layers.BatchNormalization()(x)
          x = layers.Dropout(0.5)(x)
          x = layers.Dense(64, activation = "relu", input_shape = (total_embed.shape[1], ))(total_embed)
          x = layers.BatchNormalization()(x)
          x = layers.Dropout(0.5)(x)
          x = layers.Dense(self.num_classes, activation = "softmax")(x)
          return x


    def _get_model(self):

        # Input
        word_inputs = layers.Input(shape = [], dtype = tf.string, name = "token_input")
        char_inputs = layers.Input(shape = (1, ), dtype = tf.string, name = "char_input")
        line_ids_inputs = layers.Input(shape = (self.line_ids_input_dim, ), name = "line_ids_input")
        length_lines_inputs = layers.Input(shape = (self.length_lines_input_dim, ), name = "length_lines_input")
        total_lines_inputs = layers.Input(shape = (self.total_lines_input_dim, ), name = "total_lines_input")

        #-----------------------------------------------
        # Branch outputs
        # Word-level
        word_level_output = self.word_level_branch(word_inputs)
        # print("word-level shape",word_level_output.shape)
        # # Char-level branch
        char_level_output = self.char_level_branch(char_inputs)
        # print("char-level shape",char_level_output.shape)



        concat_stats_inputs = tf.concat([line_ids_inputs, total_lines_inputs, length_lines_inputs], axis = 1)
        # #---------------------------------------------------------
        # print(concat_stats_inputs)

        concat_stats_output = self.mlp_share1(concat_stats_inputs)
        # print(concat_stats_output)

        # #Concate two embeddings
        word_char_concat = self.concatenate([word_level_output,char_level_output])

        # print("Word-char concat shape, ",word_char_concat.shape)
        # # Pass to word_char_block
        word_char_output= self.word_char_block(word_char_concat)
        # #--------------------------------------------------------
        # print(word_char_output.shape)


        # # # Concatnate 5 input
        total_embed = self.concatenate([word_char_output, concat_stats_output])
        # print(total_embed.shape)

        # # #Sequence label opt layers

        total_embed = self.sequence_opt_layer(total_embed)
        # print("Output tf shape", total_embed.shape)



        # FCN
        output_layer = self.fcn(total_embed)
        # print(output_layer.shape)
        model= tf.keras.Model(inputs=[word_inputs, char_inputs, line_ids_inputs, length_lines_inputs, total_lines_inputs],
                         outputs= output_layer,
                         name="hierarchy-BiLSTM")
        model.compile(optimizer = params.OPTIMIZER, loss = params.LOSS,
               metrics = params.METRICS)
        return model
    

    def _define_checkpoint(self):
        """
        Define checkpoint for model
        """
        if str(self.pretrained_embedding).lower() == "glove":
            if not os.path.exists(params.PENTA_BILSTM_GLOVE_MODEL_DIR):
                os.makedirs(params.PENTA_BILSTM_GLOVE_MODEL_DIR)
            checkpoint_dir = params.PENTA_BILSTM_GLOVE_MODEL_DIR
        elif str(self.pretrained_embedding).lower() == "bert":
            if not os.path.exists(params.PENTA_BILSTM_GLOVE_MODEL_DIR):
                os.makedirs(params.PENTA_BILSTM_GLOVE_MODEL_DIR)
            checkpoint_dir = params.PENTA_BILSTM_GLOVE_MODEL_DIR
        else:
            if not os.path.exists(params.PENTA_BILSTM_NOR_MODEL_DIR):
                os.makedirs(params.PENTA_BILSTM_NOR_MODEL_DIR)
            checkpoint_dir = params.PENTA_BILSTM_NOR_MODEL_DIR

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


if __name__ == "__main__":
    pass