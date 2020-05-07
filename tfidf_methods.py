from create_word_vectors import * 
import pickle 
import helper
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_embedding(X_train, X_test, word_embedding):
    print("train tfidf")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_train_tfidf_avg = average_word_vectors(X_train, word_embedding, tfidf_vectorizer, X_train_tfidf)

    print("test tfidf")
    tfidf_vectorizer = TfidfVectorizer()
    X_test_tfidf = tfidf_vectorizer.fit_transform(X_test)
    X_test_tfidf_avg = average_word_vectors(X_test, word_embedding, tfidf_vectorizer, X_test_tfidf )
    return X_train_tfidf_avg, X_test_tfidf_avg, X_train_tfidf, X_test_tfidf 

def average_word_vectors(texts ,word_embedding, tfidf_vectorizer, X_tfidf ):
    print("average word vector")
    error = 0
    avg_word_vectors = np.zeros((len(texts), len(next(iter(word_embedding.values())))))
    for i, text in enumerate(texts):
        split_text = text.split()
        nb_words = 0
        # word not found so we will assign 0 to it's features
        #not_found = [0]*avg_word_vectors.shape[1]
        for word in split_text:
            try:
                avg_word_vectors[i] += word_embedding[word.encode()] * X_tfidf[i,tfidf_vectorizer.vocabulary_[word]]
                nb_words += 1

            except KeyError: 
                continue
        if (nb_words != 0):
            avg_word_vectors[i] /= nb_words
        #print(nb_words)
    return avg_word_vectors

