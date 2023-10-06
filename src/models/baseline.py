from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import sys
import os
parent_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.append(parent_root)
from src.dataset import *

from src.config.configs import Params
param = Params()

## Naives Bayes, Logistic Regression

class BaseLine(object):
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model


    def train(self, X_train, y_train, X_val, y_val, X_test, y_test):
        model= Pipeline([
        ("vectorizer", self.vectorizer),
        ("model", self.model)
        ])
        print("Vectorizer: ", self.vectorizer)
        print("Model: ", self.model)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        print("Model accuracy on val set", model.score(X_val, y_val))
        print("Model accuracy on test set", model.score(X_test, y_test))
        return val_score, test_score
    
if __name__ == "__main__":
    print("---------------Baseline ML model------------------------")
    params = Params()
    # Define dataset
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR, num_inputs=1)

    # Define train/val/test
    train_sentences = dataset.train_sentences
    val_sentences = dataset.val_sentences
    test_sentences = dataset.test_sentences
    y_train = dataset.y_train
    y_val = dataset.y_val
    y_test = dataset.y_test
    val_score, test_score =BaseLine(vectorizer=TfidfVectorizer(), 
                                    model=LogisticRegression()).train(
                                    X_train=train_sentences, y_train=y_train,
                                    X_val=val_sentences, y_val=y_val,
                                    X_test=test_sentences, y_test=y_test)
    
    print("Baseline Val score: ", val_score)
    print("Baseline Test score: ", test_score)