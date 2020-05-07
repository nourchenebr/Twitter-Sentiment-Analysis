from kaggle_submission import * 
from run_models import * 
from preprocess import * 



print("If you want to run the models already saved in models_saving folder please enter yes, if you choose no then we will do all the preprocessing and the models training!")
user_input = input("Please enter your choice: ")
while (user_input != 'yes' and user_input != 'no'):
    print("please enter either 'yes' or 'no'")
    user_input = input("Please enter your choice: ")

if user_input == 'yes': 
    print('Your choice was Yes!')
    print('Running xgboost on saved models')
    run_xgboost()
else :
    print('Your choice was No!')
    print("Running data preprocess")
    #run_preprocess()
    print('Default values to create embedding matrix are: ngrams = 1 , pretrained = True, max_words = None')
    #X_sequences, y, Kgl_sequences, nb_word , word_embedding, glove_matrix = run_glove_embbedding(GLOVE_EMBEDDING
                                       # + "emebddings.pkl")
    print('Running models')
    run_models()
    
    print('Running xgboost on new saved models')
    run_xgboost()
    
    
print('End of the run script!')
    
