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
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_MAP = {"Penta-embedding model": "penta", "TransformerEncoder-based model": "tf_encoder"}


def get_embeddings(embedding_arg):
    embedding_arg = str(embedding_arg).lower()
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)
    embeddings = Embeddings()
    class_index = dataset.classes
    # # Word_vectorizer, word_embed
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
    Load model
    """

    model = str(MODEL_MAP[model]).lower()
    pretrained_embedding = str(pretrained_embedding).lower()
    model_dir = CHECK_POINT_MAP[model][pretrained_embedding]
    
    if model == "penta":
        loaded_model = PentaEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer,
                                        word_embed=word_embed, char_embed=char_embed,pretrained_embedding=pretrained_embedding,
                                        glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)._get_model()
        
        loaded_model.load_weights(model_dir + "/best_model.ckpt")
    else:
        loaded_model = TransformerModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, word_embed=word_embed, char_embed = char_embed,
                                num_layers=params.NUM_LAYERS, d_model=params.D_MODEL, nhead=params.N_HEAD,
                                dim_feedforward=params.DIM_FEEDFORWARD,pretrained_embedding=pretrained_embedding, glove_embed=glove_embed,
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



def main():

    st.title("Abstract Skimming Tool")
    model_options = ["Penta-embedding model","TransformerEncoder-based model"]
    embed_options = {
        "Penta-embedding model": ["None", "Glove", "BERT"],
        "TransformerEncoder-based model": ["None", "Glove", "BERT"],
    }
    model = st.sidebar.selectbox("Select your model", model_options)
    pretrained_embedding = st.sidebar.selectbox("Select embedding for {}".format(model), embed_options[model])
    word_vectorizer, char_vectorizer, word_embed, char_embed, glove_embed, bert_process, bert_layer, class_index = get_embeddings(pretrained_embedding)
    loaded_model = load_model(model=model, word_vectorizer= word_vectorizer, char_vectorizer = char_vectorizer,
                                      word_embed=word_embed, char_embed=char_embed, pretrained_embedding=pretrained_embedding,
                                      glove_embed=glove_embed, bert_process=bert_process, bert_layer = bert_layer)
    col1, col2 = st.columns(2)

    # -------------COL 1---------------------
    with col1:
        st.write('Enter your abstract: ')
        abstract = st.text_area(label='', height=800)
        predict = st.button('Extract !')
    
    
    if predict:
        with st.spinner('Wait for prediction....'):
            # --------------Extract feature-------------------------    
            list_sens = sent_tokenize(abstract)
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
            objective, background, method, conclusion, result = put_sentences_into_classes(infer_sentences=infer_sentences, preds_class=preds_class)

        with col2:
            st.markdown(f'### Objective : ')
            st.write(f'{objective}')
            st.markdown(f'### Background : ')
            st.write(f'{background}')
            st.markdown(f'### Methods : ')
            st.write(f'{method}')
            st.markdown(f'### Result : ')
            st.write(f'{result}')
            st.markdown(f'### Conclusion : ')
            st.write(f'{conclusion}')


if __name__ == "__main__":
    main()