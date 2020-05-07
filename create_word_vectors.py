import numpy as np
import os
from glove import Corpus, Glove
import helper
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle 
from paths import * 

# set max length to 30
MAX_LENGTH = 30

def load_glove_embeddings(file):
    '''
    This method returns a dict with the word followed by it's vector
    '''
    word_embedding = {}
    f = open(file,'rb')
    for line in f:
        values = line.split()
        word_embedding[values[0]] = np.array([float(x) for x in values[1:]])
    f.close()
    return word_embedding


def create_glove_matrix(word_index , nb_word, pretrained, X):
    '''
    This method creates an embedding matrix from a dict passed in argument
    '''
    X_splitted = [x.split() for x in X]
    print('Building_word_embedding')
    word_embedding = build_word_embedding(X_splitted, pretrained)
    print('building glove matrix')
    if pretrained == True:
        glove_matrix = np.zeros((nb_word + 1, 200))
        for word, i in word_index.items():
            if i > nb_word:
                continue
            embedding_vector = word_embedding.get(word.encode())
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                glove_matrix[i] = embedding_vector
        print('Matrix created')
    else:
        glove_matrix = np.zeros((nb_word + 1, 200))
        for word, i in word_index.items():
            if i > nb_word:
                continue
            embedding_vector = word_embedding(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                glove_matrix[i] = embedding_vector
    return glove_matrix, word_embedding


def append_gr(sequences, token_indice, nb_gram=2):
    '''
    This method appends n-grams to sequences.
    '''
    new_sequences = []
    for seq in sequences:
        sub_seq = seq[:]
        for i in range(len(seq[:])-nb_gram+1):
            for ng_val in range(2, nb_gram+1):
                if tuple(seq[:][i:i+ng_val]) in token_indice:
                    sub_seq.append(token_indice[tuple(seq[:][i:i+ng_val])])
        new_sequences.append(sub_seq)
    return new_sequences


def create_ngrams(train_sequences, Kgl_sequences, nb_word, maxlen = 60, n_grams = 2):
        '''
        This method creates n-grams
        '''
        print('Adding n-grams')
        uniq_ngram_set = set()
        for seq in train_sequences:
            for i in range(2, n_grams + 1):                
                uniq_ngram_set.update(set(zip(*[seq[i:] for i in range(i)])))
        token_indice = {v: k + nb_word + 1 for k, v in enumerate(uniq_ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        
        # Appending ngrams to X and X_Kaggle
        train_sequences = append_gr(train_sequences, token_indice, n_grams)
        Kgl_sequences = append_gr(Kgl_sequences, token_indice, n_grams)
        print('End adding ngrams')
        return train_sequences, Kgl_sequences, np.max(list(indice_token.keys())) + 1
    

def save_embeddings(dict, filename):
	with open(filename, "w") as f:
		for k, v in dict.items():
			line = k + str(v) + '\n'
			f.write(str(k+' '))
			for i in v:
				f.write("%s " % i)
			f.write('\n')

def create_my_glove(X_list_words):
    print("Create my glove model")
    model  = Corpus()
    print('Fitting glove model')
    model.fit(X_list_words, window = 5)
    
    glove = Glove(no_components=200, learning_rate=0.05)
    print('Fitting Embeddings model')
    glove.fit(model.matrix, epochs=50)
    glove.add_dictionary(model.dictionary)
    
    print("creating dict")
    words = {}
    for w, id_ in glove.dictionary.items():
        words[w] = np.array(glove.word_vectors[id_])
    save_embeddings(words,GLOVE_EMBEDDING +  "my_embedding.pkl")
    print("my_embedding.pkl file created")
    return words
    
    
def build_word_embedding(X_list_words, pretrained = True):
    '''
    If load is set false we will create our own glove embedding
    otherwise we will consider the glove of stanford
    '''
    if pretrained:
        print("Loading glove embedding (Stanford glove)")
        word_embedding = load_glove_embeddings(GLOVE_EMBEDDING + 'glove.twitter.27B.200d.txt')
        print("Loading ended")
        return word_embedding
    else : return create_my_glove(X_list_words)
        
def tokenize (nb_words):
    '''
    toknize with nb_words, if None consider all words
    '''
    if nb_words == None:
            tokenizer = Tokenizer(filters='')
    else:
            tokenizer = Tokenizer(nb_words=nb_words, filters='')
    return tokenizer

def create_sequences(n_grams, X, X_Kgl, max_words):
    np.random.seed(0)
    # create a tokenizer
    tokenizer = tokenize(max_words)
    # fit train words to tokenizer
    tokenizer.fit_on_texts(X)
    # get words and their indexes
    word_index = tokenizer.word_index
    # number of words
    nb_word = len(word_index)
    # transform text to sequence
    X_sequences = tokenizer.texts_to_sequences(X)
    Kgl_sequences = tokenizer.texts_to_sequences(X_Kgl)
    
    # Add n_gram 
    if n_grams != 1:
        print("Creating n_grams")
        #if n_gram is created create with 2 n_grams since tweets are not too long
        X_sequences,Kgl_sequences,nb_word = create_ngrams(X_sequences, Kgl_sequences, len(word_index), maxlen = 2*MAX_LENGTH,n_grams = 2)
    
    # Pad sequences
    X_sequences = sequence.pad_sequences(X_sequences, maxlen=MAX_LENGTH)
    Kgl_sequences = sequence.pad_sequences(Kgl_sequences, maxlen=MAX_LENGTH)

    # create y, first pos then neg
    train_size = len(X)
    y = np.array(int(train_size/2) * [0] + int(train_size/2) * [1])

    # create indices 
    indices = np.arange(X_sequences.shape[0])
    np.random.shuffle(indices)
    X_sequences = X_sequences[indices]
    y = y[indices]
    print("Sequences created!") 
    return X_sequences, Kgl_sequences, y, nb_word, word_index
    
    
    

def run_glove_embbedding(namefile, n_grams = 1, pretrained = True, max_words = None):
    print('Starting glove embedding')

    print('Loading cleaned data')
    X = pickle.load(open(CLEANED_DATA_PATH + 'X_Cleaned.pkl', "rb"))
    X_Kgl = pickle.load(open(CLEANED_DATA_PATH + 'X_Kgl_Cleaned.pkl', "rb"))
    
    print('Creating X sequence and X_kgl sequence')
    X_sequences, Kgl_sequences, y, nb_word, word_index = create_sequences(n_grams, X, X_Kgl, max_words)
    print('Creating glove matrix')
    glove_matrix, word_embedding = create_glove_matrix(word_index , nb_word, pretrained, X)
    # We save only if we have n_grams = 1 otherwise the file is too big to be saved.
    if n_grams == 1:
        print('Saving [X_sequences, y, Kgl_sequences, nb_word , word_embedding, glove_matrix] in glove_embedding folder')
        pickle.dump([X_sequences, y, Kgl_sequences, nb_word , word_embedding, glove_matrix],open(namefile, 'wb'))
    
    print("Embedding is done and files are saved")
    return X_sequences, y, Kgl_sequences, nb_word , word_embedding, glove_matrix