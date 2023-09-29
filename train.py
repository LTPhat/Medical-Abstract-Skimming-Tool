from src.config.configs import *
from src.models.baseline import *
from src.dataset import *
from src.models.attention_based_model import *
from src.models.hybrid_embeddings_model import *
from src.create_embeddings import *
from src.dataset import Dataset

dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)
train_sentences = dataset.train_sentences
val_sentences = dataset.val_sentences

train_char = dataset.train_char
val_char = dataset.val_char

y_train_one_hot = dataset.y_train_one_hot
y_val_one_hot = dataset.y_val_one_hot

word_char_dataset, word_char_val_dataset, _ = dataset._get_word_char_dataset()

embeddings = Embeddings()
word_vectorizer, word_embed = embeddings._get_word_embeddings(train_sentences)
char_vectorizer, char_embed = embeddings._get_char_embeddings(train_char)

params = Params()

# glove_txt = param.GLOVE_DIR


# glove_embed = embeddings._get_glove_embeddings(word_vectorizer, param.GLOVE_DIR)

# bert_process, bert_layer  = embeddings._get_bert_embeddings()



# hybrid_model = HybridEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer,
#                                     word_embed=word_embed, char_embed=char_embed, pretrained_embedding="bert", 
#                                     bert_process=bert_process, bert_layer=bert_layer)._get_model()

# print(hybrid_model.summary())

# model_hist = hybrid_model.fit(word_char_dataset, steps_per_epoch=int(0.1*len(word_char_dataset)),
#                               epochs=3,
#                               validation_data=word_char_val_dataset,
#                               validation_steps=int(0.1*len(word_char_val_dataset)))



# dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)
# train_sentences = dataset.train_sentences
# val_sentences = dataset.val_sentences
# test_sentences = dataset.test_sentences
# y_train = dataset.y_train
# y_val = dataset.y_val
# y_test = dataset.y_test
# baseline1 = BaseLine(TfidfVectorizer(), LogisticRegression())
# val_score, test_score = baseline1.train(train_sentences, y_train, val_sentences, y_val, test_sentences, y_test)

# print(val_score, test_score)



