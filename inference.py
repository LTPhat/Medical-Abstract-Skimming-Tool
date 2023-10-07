import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from src.config.configs import *
from src.create_embeddings import *
from src.dataset import *
from src.models.baseline import *
from src.models.attention_based import *
from src.models.transformer_encoder_based import *
from src.models.hybrid_embeddings_model import *
from src.models.penta_embeddings_model import *
from args import init_argparse, check_valid_args
from src.utils import *
import tensorflow as tf
import pandas as pd
from args import init_infer_argparse, check_valid_args
import warnings
warnings.filterwarnings("ignore")
import re



params = Params()
CHECK_POINT_MAP = {"att":{"none": params.WORD_MODEL_ATT_NOR_DIR, "glove": params.WORD_MODEL_ATT_GLOVE_DIR, "bert": params.WORD_MODEL_ATT_BERT_DIR},
                 "hybrid":{"none": params.HYBRID_NOR_MODEL_DIR, "glove": params.HYBRID_GLOVE_MODEL_DIR, "bert": params.HYBRID_BERT_MODEL_DIR}, 
                 "tf_encoder": {"none": params.TF_BASED_NOR_MODEL_DIR, "glove": params.TF_BASED_GLOVE_MODEL_DIR, "bert": params.TF_BASED_BERT_MODEL_DIR}, 
                 "penta": {"none":params.PENTA_NOR_MODEL_DIR, "glove":params.PENTA_GLOVE_MODEL_DIR, "bert": params.PENTA_BERT_MODEL_DIR}}


def read_infer_txt(infer_txt):
    with open(infer_txt, "r") as f:
        return f.readlines()


def replace_numeric_chars_with_at(list_sencentes):
    """
    Replace numeric characters with "@"
    """
    result = []
    for sent in list_sencentes:
        res = re.sub(r'\d', '@', sent)
        result.append(res)
    return result



def infer(abstract, verbose = True):
    """
    Get prediction from abstract
    args:
    - abstract: All sentences of abstract in one string.
    """
    # Init infer parser
    parser = init_infer_argparse()
    args   = parser.parse_args()

    #Check valid args
    if not check_valid_args(args):
        exit(1)

    # Sentencizer
    list_sens = sent_tokenize(abstract)

    # Store original sentence
    list_sens_org = list_sens

    #Replace numeric at @
    list_sens = replace_numeric_chars_with_at(list_sens)

    # Extract features
    line_samples = get_information_infer(list_sens)

    # Create dataframe
    infer_df = pd.DataFrame(line_samples)

    # Get features
    infer_sentences = infer_df['text']

    infer_chars = [split_into_char(line) for line in infer_sentences]
    line_ids_one_hot = tf.one_hot(infer_df['line_id'].to_numpy(), depth = params.LINE_IDS_DEPTH)

    length_lines_one_hot = tf.one_hot(infer_df['length_lines'].to_numpy(), depth = params.LENGTH_LINES_DEPTH)

    total_lines_one_hot = tf.one_hot(infer_df['total_lines'].to_numpy(), depth= params.TOTAL_LINES_DEPTH)

    # Define args variable
    model_arg = str(args.model).lower()
    embedding_arg = str(args.embedding).lower()

    embeddings = Embeddings()
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)


    # # Word_vectorizer, word_embed
    word_vectorizer, word_embed = embeddings._get_word_embeddings(dataset.train_sentences)
    char_vectorizer, char_embed = embeddings._get_char_embeddings(dataset.train_char)

    # Get type embedding
    glove_embed = embeddings._get_glove_embeddings(vectorizer=word_vectorizer, glove_txt=params.GLOVE_DIR) if str(embedding_arg).lower() == "glove" else None

    if embedding_arg == "bert":
        bert_process, bert_layer = embeddings._get_bert_embeddings()
    else:
        bert_process, bert_layer = None, None

    # Define model checkpoint dir
    model_dir = CHECK_POINT_MAP[model_arg][embedding_arg]
    if model_arg == "att":
        print("-------------Inference Attention-Based model with pretrained embedding: {}-------------------".format(embedding_arg))

        att_obj = AttentionModel(word_vectorizer=word_vectorizer, word_embed=word_embed, pretrained_embedding=embedding_arg, 
                                    glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)
        
        att_model = att_obj._get_model()
        att_model.load_weights(model_dir + "/best_model.ckpt")
        preds = att_model.predict(x = tf.constant(infer_sentences))

    elif model_arg == "hybrid":
        print("-------------Inference Hybrid model with pretrained embedding: {}-------------------".format(embedding_arg))

        hybrid_obj = HybridEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed,
                                        char_embed=char_embed, pretrained_embedding=embedding_arg,
                                        glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)
        hybrid_model = hybrid_obj._get_model()

        hybrid_model.load_weights(model_dir + "/best_model.ckpt")
        preds = hybrid_model.predict(x = tf.constant(infer_sentences, infer_chars))

    elif model_arg == "tf_encoder":
        print("-------------Inference TransformerEncoder-based with pretrained embedding: {}-------------------".format(embedding_arg))

        tf_obj = TransformerModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed, char_embed = char_embed,
                                num_layers=params.NUM_LAYERS, d_model=params.D_MODEL, nhead=params.N_HEAD,
                                dim_feedforward=params.DIM_FEEDFORWARD,pretrained_embedding=embedding_arg, glove_embed=glove_embed,
                                bert_process=bert_process, bert_layer= bert_layer)
        
        tf_model = tf_obj._get_model()
        tf_model.load_weights(model_dir + "/best_model.ckpt")


        # Make sure input has suitable data types
        infer_sentences = np.array(infer_sentences, dtype=str)
        infer_chars = np.array(infer_chars,dtype= str)
        # Get prediction
        preds = tf_model.predict(x = (infer_sentences, infer_chars, line_ids_one_hot, length_lines_one_hot, total_lines_one_hot))

    else:
        print("-------------Inference Penta-embedding model with pretrained embedding: {}-------------------".format(embedding_arg))
        penta_obj = PentaEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed, char_embed = char_embed,
                                        pretrained_embedding=embedding_arg, glove_embed=glove_embed, bert_process=bert_process, bert_layer = bert_layer)
        penta_model = penta_obj._get_model()
        penta_model.load_weights(model_dir + "/best_model.ckpt")

        # Make sure input has suitable data types
        infer_sentences = np.array(infer_sentences, dtype=str)
        infer_chars = np.array(infer_chars,dtype= str)
        # Get prediction
        preds = penta_model.predict(x = (infer_sentences, infer_chars, line_ids_one_hot, length_lines_one_hot, total_lines_one_hot))

    class_index = dataset.classes
    preds_index = np.argmax(preds, axis = 1)
    preds_class = [class_index[preds_index[i]] for i in range(0, len(preds_index))]
    
    if verbose:
        for i, sent in enumerate(list_sens_org):
            print("{} --> Pred: {} | Prob: {}".format(sent, preds_class[i], preds[i][preds_index[i]]))
    
    return preds_class


if __name__ == "__main__":
    params = Params()
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)
    infer_txt = "infer_abstract.txt"
    abstract_list = read_infer_txt(infer_txt=infer_txt)
    for i, abtract in enumerate(abstract_list):
        print("------------Predict abstract number {}--------------".format(i+1))
        preds = infer(abstract=abtract)
        print("Result:", preds)
        print()

    

