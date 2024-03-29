import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import string
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pickle


parent_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parent_root)
from src.utils import *
from src.config.configs import * 

params = Params()



class Embeddings(object):
    def __init__(self):
        # Word-level
        self.vocab_size = params.VOCAB_SIZE
        self.seq_length = params.SEQ_LENGTH
        
        # Char-level
        self.char_vocab = params.CHAR_VOCAB
        self.output_char_length = params.CHAR_LENGTH
        self.max_tokens = len(string.ascii_lowercase + string.digits + string.punctuation) + 2  # + Space, OOV

        # Positional 
        self.line_ids_depth = params.LINE_IDS_DEPTH
        self.length_lines_depth = params.LENGTH_LINES_DEPTH
        self.total_lines_depth = params.TOTAL_LINES_DEPTH

    
    def _get_word_embeddings(self, list_sentences):
        """
        Get word-level embedding layer
        args:
        - list_sentences: List of all sentences in train/val set
        return
        - word_embed: Word embedding layer
        """
        # Vectorization
        # Check if word_vectorizer obj are saved or not

            # If obj existed at disk
        if os.path.exists(params.WORD_VECTORIZATION):
            from_disk = pickle.load(open(params.WORD_VECTORIZATION, "rb"))
            print("Load pre-saved word_vectorizer object from disk at: ".format(params.WORD_VECTORIZATION))  
            word_vectorizer = TextVectorization.from_config(from_disk['config'])
            word_vectorizer.set_weights(from_disk['weights'])

        else:
            # Create folder to save vectorizer obj
            if not os.path.exists(params.VECTORIZATION):
                os.makedirs(params.VECTORIZATION)

            print("Create new word_vectorizer object ...")
            # Create new word_vectorizer obj
            word_vectorizer = TextVectorization(max_tokens=self.vocab_size, output_sequence_length=self.seq_length)
            word_vectorizer.adapt(list_sentences)

            # Save newly created obj
            pickle.dump({'config': word_vectorizer.get_config(),
             'weights': word_vectorizer.get_weights()}
            , open(params.WORD_VECTORIZATION, "wb"))

            print("Saved new word_vectorizer object to disk at: ".format(params.WORD_VECTORIZATION))
        
        word_vocab = word_vectorizer.get_vocabulary()

        print("Word vectorization on training set with vocab size: ", len(word_vocab))

        # Embedding layer
        word_embed = tf.keras.layers.Embedding(input_dim = len(word_vocab), output_dim=params.WORD_OUTPUT_DIM,
                               mask_zero=True,
                               name="word-level_embedding")
        
        
        return word_vectorizer, word_embed
    
    
    def _get_char_embeddings(self, list_char):
        """
        Get char-level embedding layer
        args:
        - list_char: List of chars split from each sentence in list_sentences
        """
        
         # If obj existed at disk
        if os.path.exists(params.CHAR_VECTORIZATION):
            from_disk = pickle.load(open(params.CHAR_VECTORIZATION, "rb"))
            print("Load pre-saved char_vectorizer object from disk at: ".format(params.CHAR_VECTORIZATION))
            char_vectorizer = TextVectorization.from_config(from_disk['config'])
            char_vectorizer.set_weights(from_disk['weights'])
        else:
            # Create folder to save vectorizer obj
            if not os.path.exists(params.VECTORIZATION):
                os.makedirs(params.VECTORIZATION)
                
            print("Create new char_vectorizer object ...")
            # Create new word_vectorizer obj
            char_vectorizer = TextVectorization(max_tokens = self.max_tokens,
                                    output_sequence_length=self.output_char_length) 
            char_vectorizer.adapt(list_char)

            # Save newly created obj
            pickle.dump({'config': char_vectorizer.get_config(),
             'weights': char_vectorizer.get_weights()}
            , open(params.CHAR_VECTORIZATION, "wb"))

        char_vocab = char_vectorizer.get_vocabulary()

        print("Char vectorization on training set with vocab size: ", len(char_vocab))

        # Embedding
        char_embed = tf.keras.layers.Embedding(input_dim = len(char_vocab), output_dim = params.CHAR_OUTPUT_DIM,
                               mask_zero=False,
                               name="character-level_embedding")

        return char_vectorizer, char_embed
    

    @staticmethod
    def create_glove_vocab(glove_txt):
        """
        Create vocab dict from glove_txt
        """

        glove_file = open(glove_txt)
        glove_embed_dict = {}
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            glove_embed_dict[word] = vector_dimensions
        glove_file.close()
        return glove_embed_dict


    @staticmethod
    def create_glove_embed_matrix(vectorizer, glove_embed_dict, embed_dim = 200):
        """
        Create glove matrix 
        args:
            - vectorizer: word_vectorizer adapted to train_set
            - glove_embed_dict: glove vocabulary dict 
        """

        corpus_vocab = vectorizer.get_vocabulary()
        vocab_size = len(corpus_vocab)
        glove_embed_matrix = np.zeros((vocab_size, embed_dim))
        for i, word in enumerate(corpus_vocab):
            word_vector = glove_embed_dict.get(word)
            if word_vector is not None:
                glove_embed_matrix[i] = word_vector
        return glove_embed_matrix
    

    def _get_glove_embeddings(self, vectorizer, glove_txt):
        """
        Get pretrained glove embedding layer
        """

        glove_embed_dict = self.create_glove_vocab(glove_txt)
        glove_embed_matrix = self.create_glove_embed_matrix(vectorizer, glove_embed_dict)
        glove_embed = layers.Embedding(input_dim=glove_embed_matrix.shape[0], output_dim=glove_embed_matrix.shape[1],
                              input_length=params.SEQ_LENGTH, trainable=False, weights=[glove_embed_matrix], name="glove_embedding")

        return glove_embed
    

    def _get_bert_embeddings(self):
        """
        Get pretrained BERT embedding layer
        """
        preprocess = hub.load(params.BERT_PROCESS_DIR)
        bert = hub.load(params.BERT_EMBED_DIR)
        preprocess_layer = hub.KerasLayer(preprocess, name='bert_input_preprocess')
        bert_layer = hub.KerasLayer(bert, name='bert_layer')

        return preprocess_layer, bert_layer

    
        
