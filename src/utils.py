import os
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st
import base64


def get_lines(txt_url):
    """
    Read data txt file 
    """
    with open(txt_url, "r") as f:
      return f.readlines()



def get_information(file_dir):
    """
    Create features from data
    Each sentence in an abstract has:
    - target: Class label
    - text: raw text
    - line id: Numeral order of a sentence in an abstract
    - line_length: Length of text
    - total_lines: Number of sentences in an abstract
    """

    input_lines = get_lines(file_dir)

    abstract_samples = []
    lines_in_one_abstract = ""
    for line in input_lines:
        # Start new abstract
        if line.startswith("###"):
            abstract_id = line[4:-1]
            one_abstract = ""
        # End an abstract
        elif line.isspace():
            lines_in_one_abstract = one_abstract.splitlines()
            for line_id, line in enumerate(lines_in_one_abstract):
                # Each dict contains infor of a line
                line_infor = {}
                line_infor['abstract_id'] = abstract_id
                # Split target and text
                line_split = line.split("\t")
                line_infor['target'] = line_split[0]
                line_infor['text'] = line_split[1].lower()
                line_infor['line_id'] = line_id
                line_infor['length_lines'] = len(line_split[1].split(" ")) # Num of words in a line
                line_infor['total_lines'] = len(lines_in_one_abstract) - 1 # Num of lines in an abstract
                abstract_samples.append(line_infor)
        else:
            one_abstract += line
    return abstract_samples


def get_information_infer(samples):
    """
    Extract feature in inference phase
    """

    total_line = len(samples)

    sample_lines = []
    for id, line in enumerate(samples):
        one_line = {}
        one_line['text'] = str(line)
        one_line['line_id'] = id
        one_line['length_lines'] = len(line.split(" "))
        one_line['total_lines'] = total_line - 1
        sample_lines.append(one_line)
    return sample_lines



def split_into_char(line):
    """Split a line into char"""
    return " ".join(list(line))



def convert_to_one_hot(y_pred):
    """
    Convert probability vector y_pred into one-hot vector
    """
    for row in range(len(y_pred)):
        max_index = np.argmax(y_pred[row])
        y_pred[row] = np.zeros((1, len(y_pred[row])))
        y_pred[row][max_index] = 1
    return y_pred



def convert_to_one_hot(y_pred):
    """
    Convert probability vector into one-hot vector
    """
    for row in range(len(y_pred)):
        max_index = np.argmax(y_pred[row])
        y_pred[row] = np.zeros((1, len(y_pred[row])))
        y_pred[row][max_index] = 1
    return y_pred



def get_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Get confusion matrix of model predictio
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, square=True,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()



# ---------------------Streamlit App utils------------------------------

# Decode image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Set background for local web
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def get_block(custom_text, back_ground):
    block = f"""
        <div style="
                    {back_ground}
                    border-radius: 6px;
                    min-height: 80px;
                    padding: 20px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    border-radius: 25px;
                    box-sizing: border-box;
                    text-align: justify;
                    color: transparent;
                    ">
                    <h4 style="color:#FFFFFFF; font-family: cursive; font-size: 16px; font-weight: 200">
                            {custom_text}</h4>
                    </div>
                    </br>"""
    st.markdown(block, unsafe_allow_html=True)