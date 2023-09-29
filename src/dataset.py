import sys
from config.configs import *
import pandas as pd
from utils import get_information, split_into_char
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)
from config.configs import *
params = Params()




class Dataset(object):
    def __init__(self, train_txt, val_txt, test_txt, num_inputs = 5):
        params = Params()
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.test_txt = test_txt
        self.classes = None
        self.batch_size = params.BATCH_SIZE

        # Data attributes
        self.train_samples, self.val_samples, self.test_samples = self._get_samples()

        # Data dataframe 
        self.train_df, self.val_df, self.test_df = self._get_dataframe()

        # Word-level input
        self.train_sentences, self.val_sentences, self.test_sentences = self._get_word_sentences()

        # Char-level input
        self.train_char, self.val_char, self.test_char = self._get_char_sentences() 

        # Positional input
            ## Line_ids 
        self.line_ids_one_hot, self.line_ids_val_one_hot, self.line_ids_test_one_hot = self._get_lines_id() if num_inputs == 5 else None

            ## Length_lines
        self.length_lines_one_hot, self.length_lines_val_one_hot, self.length_lines_test_one_hot = self._get_length_lines() if num_inputs == 5 else None

            ## Total_lengths
        self.total_lines_one_hot, self.total_lines_val_one_hot, self.total_lines_test_one_hot = self._get_total_lines() if num_inputs == 5 else None

        # Label one-hot
        self.y_train_one_hot, self.y_val_one_hot, self.y_test_one_hot = self._one_hot_encoder()
        
        # Label indexes
        self.y_train, self.y_val, self.y_test = self._label_encoder()


    def _get_samples(self):
        """
        Get data features
        """
        self.train_samples = get_information(self.train_txt)
        self.val_samples = get_information(self.val_txt)
        self.test_samples = get_information(self.test_txt)

        return self.train_samples, self.val_samples, self.test_samples
    


    def _get_dataframe(self):
        """
        Create dataframe for visualization from samples
        """
        self.train_df = pd.DataFrame(self.train_samples)
        self.val_df = pd.DataFrame(self.val_samples)
        self.test_df = pd.DataFrame(self.test_samples)
        return self.train_df, self.val_df, self.test_df



    def _get_word_sentences(self):
        """
        Get list of sentences of from train/val/test set  (Word-level tokens)
        """
        self.train_sentences = self.train_df['text']
        self.val_sentences = self.val_df['text']
        self.test_sentences = self.test_df['text']
        return self.train_sentences, self.val_sentences, self.test_sentences


    def _get_char_sentences(self):
        """
        Get list of chars of sentences from train/val/test set (Char-level tokens)
        """

        self.train_char = [split_into_char(line) for line in self.train_df['text']]
        self.val_char = [split_into_char(line) for line in self.val_df['text']]
        self.test_char = [split_into_char(line) for line in self.test_df['text']]

        return self.train_char, self.val_char, self.test_char
    


    def _one_hot_encoder(self):
        """
        Get one-hot label vector from labels
        """
        one_hot_encoder = OneHotEncoder(sparse=False)
        self.y_train_one_hot = one_hot_encoder.fit_transform(self.train_df["target"].to_numpy().reshape(-1, 1))
        self.y_val_one_hot = one_hot_encoder.fit_transform(self.val_df["target"].to_numpy().reshape(-1, 1))
        self.y_test_one_hot = one_hot_encoder.fit_transform(self.test_df["target"].to_numpy().reshape(-1, 1))
        return self.y_train_one_hot, self.y_val_one_hot, self.y_test_one_hot



    def _label_encoder(self) :
        """
        Get index class from labels
        """
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.train_df["target"].to_numpy())
        self.y_val = label_encoder.transform(self.val_df["target"].to_numpy())
        self.y_test = label_encoder.transform(self.test_df["target"].to_numpy())
        return self.y_train, self.y_val, self.y_test



#-----------------------POSITIONAL FEATURES------------------------------------------
    def _get_lines_id(self):
        """
        Get line_ids feature
        """
        try:
            self.line_ids_one_hot = tf.one_hot(self.train_df['line_id'].to_numpy(), depth = params.LINE_IDS_DEPTH)
            self.line_ids_val_one_hot = tf.one_hot(self.val_df['line_id'].to_numpy(), depth = params.LINE_IDS_DEPTH)
            self.line_ids_test_one_hot = tf.one_hot(self.test_df['line_id'].to_numpy(), depth = params.LINE_IDS_DEPTH)
        except Exception as e:
            print("Error encoding line_ids:", e)
        return self.line_ids_one_hot, self.line_ids_val_one_hot, self.line_ids_test_one_hot



    def _get_length_lines(self):
        try:
            self.length_lines_one_hot = tf.one_hot(self.train_df['length_lines'].to_numpy(), depth = params.LENGTH_LINES_DEPTH)
            self.length_lines_val_one_hot = tf.one_hot(self.val_df['length_lines'].to_numpy(), depth = params.LENGTH_LINES_DEPTH)
            self.length_lines_test_one_hot = tf.one_hot(self.test_df['length_lines'].to_numpy(), depth = params.LENGTH_LINES_DEPTH)
        except Exception as e:
            print("Error encoding length_lines:", e)
        return self.length_lines_one_hot, self.length_lines_val_one_hot, self.length_lines_test_one_hot



    def _get_total_lines(self):
        try:
            self.total_lines_one_hot = tf.one_hot(self.train_df['total_lines'].to_numpy(), depth = params.TOTAL_LINES_DEPTH)
            self.total_lines_val_one_hot = tf.one_hot(self.val_df['total_lines'].to_numpy(), depth = params.TOTAL_LINES_DEPTH)
            self.total_lines_test_one_hot = tf.one_hot(self.test_df['total_lines'].to_numpy(), depth = params.TOTAL_LINES_DEPTH)
        except Exception as e:
            print("Error encoding line_ids:", e)
        return self.total_lines_one_hot, self.total_lines_val_one_hot, self.total_lines_test_one_hot
# ----------------------------------------------------------------------------------------


# -------------------CREATE DATASET------------------------------------------------------
    def _get_word_dataset(self):
        """
        Get dataset with 1 input: Word-level token
        .... 
        """
        # Train set
        word_input_data = tf.data.Dataset.from_tensor_slices(self.train_sentences)
        word_input_label = tf.data.Dataset.from_tensor_slices(self.y_train_one_hot)

        word_train_dataset = tf.data.Dataset.zip((word_input_data, word_input_label))
        word_train_dataset = word_train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Val set
        word_val_data = tf.data.Dataset.from_tensor_slices(self.val_sentences)
        word_val_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)

        word_val_dataset = tf.data.Dataset.zip((word_val_data, word_val_label))
        word_val_dataset = word_val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Test_set 
        word_test_data = tf.data.Dataset.from_tensor_slices(self.test_sentences)
        word_test_label = tf.data.Dataset.from_tensor_slices(self.y_test_one_hot)

        word_test_dataset = tf.data.Dataset.zip((word_test_data, word_test_label))
        word_test_dataset = word_test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return word_train_dataset, word_val_dataset, word_test_dataset
    


    def _get_word_char_dataset(self):
        """
        Get dataset with hybrid inputs: Word-level tokens and char-level token
        """

        # Train set
        word_char_data = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_char)) 
        word_char_label = tf.data.Dataset.from_tensor_slices(self.y_train_one_hot) 
        word_char_dataset = tf.data.Dataset.zip((word_char_data, word_char_label)) 
        word_char_dataset = word_char_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Val set
        word_char_val_data = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_char)) 
        word_char_val_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)
        word_char_val_dataset = tf.data.Dataset.zip((word_char_val_data, word_char_val_label)) 
        word_char_val_dataset = word_char_val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        #Test set
        word_char_test_data = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_char)) 
        word_char_test_label = tf.data.Dataset.from_tensor_slices(self.y_test_one_hot)
        word_char_test_dataset = tf.data.Dataset.zip((word_char_test_data, word_char_test_label)) 
        word_char_test_dataset = word_char_test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return word_char_dataset, word_char_val_dataset, word_char_test_dataset
    

    def _get_tetra_dataset(self):
        """
        Get tetra inputs dataset: word-level tokens, char-level tokens, lines_ids tokens, total_lines tokens
        """
        # Train set
        tetra_data = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_char,
                                                            self.line_ids_one_hot, self.total_lines_one_hot))
        tetra_label = tf.data.Dataset.from_tensor_slices(self.y_train_one_hot)

        tetra_train_dataset = tf.data.Dataset.zip((tetra_data, tetra_label))
        tetra_train_dataset = tetra_train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        # Val set
        tetra_val_data = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_char,
                                                            self.line_ids_val_one_hot, self.total_lines_val_one_hot))
        tetra_val_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)
        tetra_val_dataset = tf.data.Dataset.zip((tetra_val_data, tetra_val_label))
        tetra_val_dataset = tetra_val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        # Test set
        tetra_test_data = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_char,
                                                            self.line_ids_test_one_hot, self.total_lines_test_one_hot))
        tetra_test_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)
        tetra_test_dataset = tf.data.Dataset.zip((tetra_test_data, tetra_test_label))
        tetra_test_dataset = tetra_test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        return tetra_train_dataset, tetra_val_dataset, tetra_test_dataset



    def _get_penta_dataset(self):
        """
        Get penta inputs dataset: Word-level tokens, char-level tokens, line_ids tokens, length_lines tokens, total_lines tokens
        """
        # Train set
        penta_data = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_char,
                                                            self.line_ids_one_hot, self.length_lines_one_hot, self.total_lines_one_hot))
        penta_label = tf.data.Dataset.from_tensor_slices(self.y_train_one_hot)

        penta_train_dataset = tf.data.Dataset.zip((penta_data, penta_label))
        penta_train_dataset = penta_train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        # Val set
        penta_val_data = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_char,
                                                            self.line_ids_val_one_hot, self.length_lines_val_one_hot, self.total_lines_val_one_hot))
        penta_val_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)
        penta_val_dataset = tf.data.Dataset.zip((penta_val_data, penta_val_label))
        penta_val_dataset = penta_val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        # Test set
        penta_test_data = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_char,
                                                            self.line_ids_test_one_hot, self.length_lines_test_one_hot, self.total_lines_test_one_hot))
        penta_test_label = tf.data.Dataset.from_tensor_slices(self.y_val_one_hot)
        penta_test_dataset = tf.data.Dataset.zip((penta_test_data, penta_test_label))
        penta_test_dataset = penta_test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        return penta_train_dataset, penta_val_dataset, penta_test_dataset
    

if __name__ == "__main__":
    params = Params()
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)

    word_char_train, word_char_val,_ = dataset._get_word_char_dataset()
    print("Done")