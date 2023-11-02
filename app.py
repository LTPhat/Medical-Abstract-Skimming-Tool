import streamlit as st
import streamlit.components.v1 as components
import base64
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from inference import *
from src.config.configs import Params

params = Params()

st.set_page_config(
    page_title="Abstract Skimming Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
#### ADD BILSTM MODEL

MODEL_MAP = {"Penta-embedding model": "penta", "TransformerEncoder-based model": "tf_encoder", "Hierarchy-BiLSTM": "bilstm"}

# Colors for prediction
BACK_GROUNDS = ["background: rgb(0,0,0); background: linear-gradient(29deg, rgba(0,0,0,1) 75%, rgba(213,0,0,1) 95%);",
               "background: rgb(0,0,0); background: linear-gradient(29deg, rgba(0,0,0,1) 75%, rgba(0,183,213,1) 95%);",
               "background: rgb(0,0,0); background: linear-gradient(29deg, rgba(0,0,0,1) 75%, rgba(213,164,0,1) 95%);",
               "background: rgb(0,0,0); background: linear-gradient(29deg, rgba(0,0,0,1) 75%, rgba(54,213,0,1) 95%);",
               "background: rgb(0,0,0); background: linear-gradient(29deg, rgba(0,0,0,1) 75%, rgba(125,0,213,1) 95%);",
               ]


def get_embeddings(embedding_arg):
    """
    Load embeddings from embedding_arg
    """

    embedding_arg = str(embedding_arg).lower()
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)
    embeddings = Embeddings()
    class_index = dataset.classes

    # Word_vectorizer, word_embed
    word_vectorizer, word_embed = embeddings._get_word_embeddings(dataset.train_sentences)
    char_vectorizer, char_embed = embeddings._get_char_embeddings(dataset.train_char)

    # Get type embedding
    glove_embed = embeddings._get_glove_embeddings(vectorizer=word_vectorizer, glove_txt=params.GLOVE_DIR) if str(embedding_arg).lower() == "glove" else None

    if embedding_arg == "bert":
        bert_process, bert_layer = embeddings._get_bert_embeddings()
    else:
        bert_process, bert_layer = None, None
    return word_vectorizer, char_vectorizer, word_embed, char_embed, glove_embed, bert_process, bert_layer, class_index




def load_model(model, word_vectorizer, char_vectorizer, word_embed, char_embed, pretrained_embedding,
               glove_embed, bert_process, bert_layer):
    """
    Load model from user's request
    """

    model = str(MODEL_MAP[model]).lower()
    pretrained_embedding = str(pretrained_embedding).lower()
    model_dir = CHECK_POINT_MAP[model][pretrained_embedding]
    
    if model == "penta":
        loaded_model = PentaEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer,
                                        word_embed=word_embed, char_embed=char_embed,pretrained_embedding=pretrained_embedding,
                                        glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)._get_model()
        
        loaded_model.load_weights(model_dir + "/best_model.ckpt")
        
    elif model == "tf_encoder":
        loaded_model = TransformerModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed, char_embed = char_embed,
                                num_layers=params.NUM_LAYERS, d_model=params.D_MODEL, nhead=params.N_HEAD,
                                dim_feedforward=params.DIM_FEEDFORWARD,pretrained_embedding=pretrained_embedding, glove_embed=glove_embed,
                                bert_process=bert_process, bert_layer= bert_layer)._get_model()
        
        model_dir = CHECK_POINT_MAP[model][pretrained_embedding]
        loaded_model.load_weights(model_dir + "/best_model.ckpt")
    
    else:
        loaded_model = HierarchyBiLSTM(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed, char_embed = char_embed,
                                pretrained_embedding=pretrained_embedding, glove_embed=glove_embed,
                                bert_process=bert_process, bert_layer= bert_layer)._get_model()
        
        model_dir = CHECK_POINT_MAP[model][pretrained_embedding]
        loaded_model.load_weights(model_dir + "/best_model.ckpt")
    
    return loaded_model



def put_sentences_into_classes(infer_sentences, preds_class):
    """
    Separate infer sentences into its own predicted classes
    """

    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    for line, pred in zip(infer_sentences, preds_class):
        if pred == 'OBJECTIVE':
            objective = objective + " " +line
        
        elif pred == 'BACKGROUND':
            background = background + " " +line
        
        elif pred == 'METHODS':
            method = method + " " +line
        
        elif pred == 'RESULTS':
            result = result + " " +line
        
        elif pred == 'CONCLUSIONS':
            conclusion = conclusion + " " + line
        else: 
            raise NameError("There is something wrong while predicting...")
        
    return objective, background, method, conclusion, result



@st.cache_data(experimental_allow_widgets=True)
def pre_load():
    """
    Pre-load some element when loading app
    """
    st.markdown("<h1 style='text-align: center; color:#f7cf25; font-family: cursive; padding: 20px; font-size: 40px; '>Abstract Skimming Tool</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-family: cursive; padding: 0px; margin: 20px; margin-bottom: 30px; font-size: 20px; '>An NLP model enables researchers to skim paper abstracts and extract information better.</h3>",
                unsafe_allow_html=True)
    st.write()
    st.write()

    # Selectbox 1
    model_options = ["Penta-embedding model","TransformerEncoder-based model", "Hierarchy-BiLSTM"]
    
    # Selectbox 2
    embed_options = {
        "Penta-embedding model": ["None", "Glove", "BERT"],
        "TransformerEncoder-based model": ["None", "Glove", "BERT"],
        "Hierarchy-BiLSTM model": ["Glove", "BERT"],
    }
    # Get option from user
    model = st.sidebar.selectbox("Select your model", model_options)
    pretrained_embedding = st.sidebar.selectbox("Select embedding for {}".format(model), embed_options[model])

    # Load properties from user's option
    word_vectorizer, char_vectorizer, word_embed, char_embed, glove_embed, bert_process, bert_layer, class_index = get_embeddings(pretrained_embedding)
    loaded_model = load_model(model=model, word_vectorizer= word_vectorizer, char_vectorizer = char_vectorizer,
                                      word_embed=word_embed, char_embed=char_embed, pretrained_embedding=pretrained_embedding,
                                      glove_embed=glove_embed, bert_process=bert_process, bert_layer = bert_layer)

    return loaded_model, class_index



def get_prediction(loaded_model, class_index):
    """
    Get prediction
    """

    col1, col2 = st.columns(2)

    # -------------COL 1---------------------
    with col1:
        st.write('Enter your abstract: ')
        abstract = st.text_area(label='', height=400)
        predict = st.button('Extract')
    
    
    if predict:
        with st.spinner('Wait for prediction....'):
            # --------------Extract feature-------------------------
            
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

            # Str type
            infer_sentences = np.array(infer_sentences, dtype=str)
            infer_chars = np.array(infer_chars,dtype= str)

            line_ids_one_hot = tf.one_hot(infer_df['line_id'].to_numpy(), depth = params.LINE_IDS_DEPTH)

            length_lines_one_hot = tf.one_hot(infer_df['length_lines'].to_numpy(), depth = params.LENGTH_LINES_DEPTH)

            total_lines_one_hot = tf.one_hot(infer_df['total_lines'].to_numpy(), depth= params.TOTAL_LINES_DEPTH)

        
            preds = loaded_model.predict(x = (infer_sentences, infer_chars, line_ids_one_hot, length_lines_one_hot, total_lines_one_hot))
            preds_index = np.argmax(preds, axis = 1)

            # Get label
            preds_class = [class_index[preds_index[i]] for i in range(0, len(preds_index))]
            objective, background, method, conclusion, result = put_sentences_into_classes(infer_sentences=list_sens_org, preds_class=preds_class)

        with col2:
            st.markdown(f'### Objective: ')
            get_block(objective,BACK_GROUNDS[0])
            st.markdown(f'### Background: ')
            get_block(background,BACK_GROUNDS[1])
            st.markdown(f'### Method: ')
            get_block(method, BACK_GROUNDS[2])
            st.markdown(f'### Result: ')
            get_block(result, BACK_GROUNDS[3])
            st.markdown(f'### Conclusion: ')
            get_block(conclusion, BACK_GROUNDS[4])


if __name__ == "__main__":
    set_background("images/bg3.jpg")
    loaded_model, class_index = pre_load()
    get_prediction(loaded_model=loaded_model, class_index=class_index)