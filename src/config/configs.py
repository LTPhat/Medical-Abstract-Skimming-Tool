import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)


class Params(object):
    def __init__(self):
        # Directory params
        self.DATA_DIR =  project_root +"/data/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign"
        self.TRAIN_DIR = self.DATA_DIR + '/train.txt'
        self.VAL_DIR = self.DATA_DIR + '/dev.txt'
        self.TEST_DIR = self.DATA_DIR + '/test.txt'
        self.CHECK_POINT_DIR =  "/checkpoints"


        # Numerical params
        self.BATCH_SIZE = 32
        self.NUM_CLASSES = 5

        # Model params
        self.OPTIMIZER = tf.keras.optimizers.Adam()
        self.EPOCHS = 3
        self.LOSS = tf.keras.losses.CategoricalCrossentropy()
        self.METRICS = tf.keras.metrics.CategoricalAccuracy()
        self.MONITOR = "val_categorical_accuracy"

        # Vectorizer and Embedding params

        #=== WORD_LEVEL====
        self.VOCAB_SIZE = 68000             # Word-level vocab size 
        self.SEQ_LENGTH = 55                # 95% percentile num of words of a sentence
        self.WORD_OUTPUT_DIM = 128          # Word embedding vectors dim


        #=== CHAR_LEVEL====
        self.CHAR_VOCAB = 28                # Char-level vocab size
        self.CHAR_LENGTH = 290              # 95% percentile num of chars in a sentence
        self.CHAR_OUTPUT_DIM = 25

        # Feature params
        self.LINE_IDS_DEPTH = 15            # 98% percentile value of line_ids feature
        self.TOTAL_LINES_DEPTH = 20         # 97% percentile value of total_lines feature
        self.LENGTH_LINES_DEPTH = 55        # 95% percentile value of length_lines feature

        # GLOVE EMBEDDING
        self.GLOVE_DIR = project_root + "/glove/glove.6B.200d.txt"

        # BERT_MODEL 
        self.BERT_PROCESS_DIR = project_root + "/bert/bert_en_uncased_preprocess_3"       
        self.BERT_EMBED_DIR =  project_root + "/bert/experts_bert_pubmed_2"                # Pretrained BERT layer

        # TRANSFORMER MODEL PARAMS
        self.NUM_LAYERS = 4
        self.N_HEAD = 8
        self.DIM_FEEDFORWARD = 256
        self.D_MODEL = 128

        # CHECKPOINT DIR

        ## penta-model
        self.PENTA_NOR_MODEL_DIR = project_root + "/checkpoints/penta_model/penta_embedding/nor_model"
        self.PENTA_BERT_MODEL_DIR = project_root + "/checkpoints/penta_model/penta_embedding/bert_model"
        self.PENTA_GLOVE_MODEL_DIR = project_root + "/checkpoints/penta_model/penta_embedding/glove_model"

        self.PENTA_BILSTM_NOR_MODEL_DIR = project_root + "/checkpoints/penta_model/hierarchy_BiLSTM/nor_model"
        self.PENTA_BILSTM_GLOVE_MODEL_DIR = project_root + "/checkpoints/penta_model/hierarchy_BiLSTM/glove_model"
        self.PENTA_BILSTM_BERT_MODEL_DIR = project_root + "/checkpoints/penta_model/hierarchy_BiLSTM/bert_model"

        self.TF_BASED_NOR_MODEL_DIR = project_root + "/checkpoints/penta_model/transformer_model/nor_model"
        self.TF_BASED_GLOVE_MODEL_DIR = project_root + "/checkpoints/penta_model/transformer_model/glove_model"
        self.TF_BASED_BERT_MODEL_DIR = project_root + "/checkpoints/penta_model/transformer_model/bert_model"

        ## Hybrid model
        self.HYBRID_NOR_MODEL_DIR = project_root + "/checkpoints/hybrid_model/nor_model"
        self.HYBRID_GLOVE_MODEL_DIR = project_root + "/checkpoints/hybrid_model/glove_model"
        self.HYBRID_BERT_MODEL_DIR = project_root + "/checkpoints/hybrid_model/bert_model"


        # Test-result dir
        self.RESULT_DIR = project_root + "/results.txt"

        # Saved vectorization object pkl
        self.VECTORIZATION = project_root + "/text_vectorization_obj/"
        self.WORD_VECTORIZATION = project_root + "/text_vectorization_obj/tv_layer.pkl"
        self.CHAR_VECTORIZATION = project_root + "/text_vectorization_obj/char_tv_layer.pkl"


if __name__ == "__main__":
    params = Params()
    print(os.path.exists(params.DATA_DIR))
    print(os.path.exists(params.RESULT_DIR))



