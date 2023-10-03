from src.config.configs import *
from src.create_embeddings import *
from src.dataset import *
from src.models.baseline import *
from src.models.attention_based import *
from src.models.transformer_encoder_based import *
from src.models.hybrid_embeddings_model import *
from src.models.penta_embeddings_model import *
from args import init_argparse, check_valid_args
import time



def main():
    # Define config params
    params = Params()

    # Define Parser
    parser = init_argparse()
    args   = parser.parse_args()

    #Check valid args
    if not check_valid_args(args):
        exit(1)

    # Define args variable
    model_arg = args.model
    embedding_arg = args.embedding
    dataset_size = float(args.dataset_size)
    params.BATCH_SIZE = int(args.batch_size)
    params.EPOCHS = int(args.epochs)

    # Define dataset obj
    dataset = Dataset(train_txt=params.TRAIN_DIR, val_txt=params.VAL_DIR, test_txt=params.TEST_DIR)

    # Define Embedding 
    embeddings = Embeddings()

    # Word_input
    train_sentences = dataset.train_sentences
    val_sentences = dataset.val_sentences
    test_sentences = dataset.test_sentences

    # Word_vectorizer, word_embed
    word_vectorizer, word_embed = embeddings._get_word_embeddings(train_sentences)



    if str(model_arg).lower() == "att":
        print("-------------Training Attention-Based model-------------------")

        # Word-input dataset
        word_dataset, word_val_dataset, word_test_dataset = dataset._get_word_dataset()
        att_obj = AttentionModel(word_vectorizer=word_vectorizer, word_embed=word_embed, pretrained_embedding=embedding_arg)
        att_model = att_obj._get_model()
        att_checkpoint = att_obj._define_checkpoint()

        # Start measuring time
        start_time = time.time()
        att_hist = att_model.fit(word_dataset, steps_per_epoch=int(dataset_size*len(word_dataset)),
                              epochs=params.EPOCHS,
                              validation_data=word_val_dataset,
                              validation_steps=int(dataset_size*len(word_val_dataset)),
                              callbacks = [att_checkpoint])
        print("-------------Training Attention-Based model completed! -------------------")
        end_time = time.time()

        # Get training time
        training_time = end_time - start_time

        # Evaluation
        print("-------------Evaluate on validation set -------------------")
        val_metrics = att_model.evaluate(word_val_dataset)
        print("-------------Evaluate on test set -------------------")
        test_metrics = att_model.evaluate(word_test_dataset)
        
        # Write results
        with open(params.RESULT_DIR, 'a') as file:
            file.write("Metrics on Attention-based model: \n")
            file.write('Val loss: {} | Val accuracy: {}\n'.format(val_metrics[0], val_metrics[1]))
            file.write('Test loss: {} | Test accuracy: {}\n'.format(test_metrics[0], test_metrics[1]))
            file.write('Time training: {} s\n'.format(training_time))
        file.close()
        print("Writing result completed! Check at results.txt.")



    elif str(model_arg).lower() == "hybrid":
        print("-------------Training Hybrid-embedding model-------------------")

        # Addtional char-vectorizer, char_embed
        train_char = dataset.train_char
        char_vectorizer, char_embed = embeddings._get_char_embeddings(train_char)
        
        # Get type embedding
        glove_embed = embeddings._get_glove_embeddings(vectorizer=word_vectorizer, glove_txt=params.GLOVE_DIR) if str(embedding_arg).lower() == "glove" else None
        bert_process, bert_layer = embeddings._get_bert_embeddings() if str(embedding_arg).lower() == "bert" else None, None

        # Get word-char dataset
        word_char_dataset, word_char_val_dataset, word_char_test_dataset = dataset._get_word_char_dataset()
        hybrid_obj = HybridEmbeddingModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer,
                                          word_embed=word_embed, char_embed=char_embed, pretrained_embedding=embedding_arg,
                                          glove_embed=glove_embed, bert_process=bert_process, bert_layer= bert_layer)
        
        hybrid_model = hybrid_obj._get_model()
        hybrid_checkpoint = hybrid_obj._define_checkpoint()

        # Start measuring time
        start_time = time.time()
        hybrid_hist = hybrid_model.fit(word_char_dataset, steps_per_epoch=int(dataset_size*len(word_char_dataset)),
                              epochs=params.EPOCHS,
                              validation_data=word_char_val_dataset,
                              validation_steps=int(dataset_size*len(word_char_val_dataset)),
                              callbacks = [hybrid_checkpoint])
        print("-------------Training Hybrid-embedding model completed! -------------------")
        end_time = time.time()

        # Get training time
        training_time = end_time - start_time

        # Evaluation
        print("-------------Evaluate on validation set -------------------")
        val_metrics = hybrid_model.evaluate(word_char_val_dataset)
        print("-------------Evaluate on test set -------------------")
        test_metrics = hybrid_model.evaluate(word_char_test_dataset)
        
        # Write results
        with open(params.RESULT_DIR, 'a') as file:
            file.write("Metrics on Hybrid-embeding model: \n")
            file.write('Val loss: {} | Val accuracy: {}\n'.format(val_metrics[0], val_metrics[1]))
            file.write('Test loss: {} | Test accuracy: {}\n'.format(test_metrics[0], test_metrics[1]))
            file.write('Time training: {} s\n'.format(training_time))
        file.close()
        print("Writing result completed! Check at results.txt.")


    elif str(model_arg).lower() == "tf_encoder":
        print("-------------Training TransformerEncoder-based model-------------------")

        # Addtional char-vectorizer, char_embed
        train_char = dataset.train_char
        char_vectorizer, char_embed = embeddings._get_char_embeddings(train_char)

        # Get type embedding
        glove_embed = embeddings._get_glove_embeddings(vectorizer=word_vectorizer, glove_txt=params.GLOVE_DIR) if str(embedding_arg).lower() == "glove" else None
        bert_process, bert_layer = embeddings._get_bert_embeddings() if str(embedding_arg).lower() == "bert" else None, None

        # Get penta-dataset
        penta_dataset, penta_val_dataset, penta_test_dataset = dataset._get_penta_dataset()
        tf_obj = TransformerModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, 
                                    word_embed=word_embed, char_embed=char_embed, num_layers=params.NUM_LAYERS, 
                                    d_model=params.D_MODEL, nhead=params.N_HEAD, dim_feedforward=params.DIM_FEEDFORWARD,
                                    pretrained_embedding=embedding_arg, glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)
        tf_model = tf_obj._get_model()

        tf_checkpoint = tf_obj._define_checkpoint()

        # Start measuring time
        start_time = time.time()
        tf_hist = tf_model.fit(penta_dataset, steps_per_epoch=int(dataset_size*len(penta_dataset)),
                              epochs=params.EPOCHS,
                              validation_data=penta_val_dataset,
                              validation_steps=int(dataset_size*len(penta_val_dataset)),
                              callbacks = [tf_checkpoint])
        print("-------------Training TransformerEncoder-based model completed!---------------------")
        end_time = time.time()

        # Get training time
        training_time = end_time - start_time

        # Evaluation
        print("-------------Evaluate on validation set -------------------")
        val_metrics = tf_model.evaluate(penta_val_dataset)
        print("-------------Evaluate on test set -------------------")
        test_metrics = tf_model.evaluate(penta_test_dataset)
        
        # Write results
        with open(params.RESULT_DIR, 'a') as file:
            file.write("Metrics on TransformerEncoder-based model: \n")
            file.write('Val loss: {} | Val accuracy: {}\n'.format(val_metrics[0], val_metrics[1]))
            file.write('Test loss: {} | Test accuracy: {}\n'.format(test_metrics[0], test_metrics[1]))
            file.write('Time training: {} s\n'.format(training_time))
        file.close()
        print("Writing result completed! Check at results.txt.")
    

    else:

        print("-------------Training Penta-embedding model-------------------")

        # Addtional char-vectorizer, char_embed
        train_char = dataset.train_char
        char_vectorizer, char_embed = embeddings._get_char_embeddings(train_char)

        # Get type embedding
        glove_embed = embeddings._get_glove_embeddings(vectorizer=word_vectorizer, glove_txt=params.GLOVE_DIR) if str(embedding_arg).lower() == "glove" else None
        bert_process, bert_layer = embeddings._get_bert_embeddings() if str(embedding_arg).lower() == "bert" else None, None

        # Get penta-dataset
        penta_dataset, penta_val_dataset, penta_test_dataset = dataset._get_penta_dataset()
        
        penta_obj = TransformerModel(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer, 
                                    word_embed=word_embed, char_embed=char_embed, num_layers=params.NUM_LAYERS, 
                                    d_model=params.D_MODEL, nhead=params.N_HEAD, dim_feedforward=params.DIM_FEEDFORWARD,
                                    pretrained_embedding=embedding_arg, glove_embed=glove_embed, bert_process=bert_process, bert_layer=bert_layer)
        penta_model = penta_obj._get_model()

        penta_checkpoint = penta_obj._define_checkpoint()

        # Start measuring time
        start_time = time.time()
        penta_hist = penta_model.fit(penta_dataset, steps_per_epoch=int(dataset_size*len(penta_dataset)),
                              epochs=params.EPOCHS,
                              validation_data=penta_val_dataset,
                              validation_steps=int(dataset_size*len(penta_val_dataset)),
                              callbacks = [penta_checkpoint])
        print("-------------Training Penta-embedding model completed!---------------------")
        end_time = time.time()

        # Get training time
        training_time = end_time - start_time

        # Evaluation
        print("-------------Evaluate on validation set -------------------")
        val_metrics = penta_model.evaluate(penta_val_dataset)
        print("-------------Evaluate on test set -------------------")
        test_metrics = penta_model.evaluate(penta_test_dataset)
        
        # Write results
        with open(params.RESULT_DIR, 'a') as file:
            file.write("Metrics on Penta-embedding model: \n")
            file.write('Val loss: {} | Val accuracy: {}\n'.format(val_metrics[0], val_metrics[1]))
            file.write('Test loss: {} | Test accuracy: {}\n'.format(test_metrics[0], test_metrics[1]))
            file.write('Time training: {} s\n'.format(training_time))
        file.close()
        print("Writing result completed! Check at results.txt.")

if __name__ == "__main__":
    main()