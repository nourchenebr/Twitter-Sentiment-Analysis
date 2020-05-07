# import necessary libraries
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import enchant
import itertools
import wordninja
import pickle
from paths import * 

# Loading stopwords list from NLTK
stoplist = set(stopwords.words("english"))

## Remove words that denote sentiment
for w in ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn', 'shouldn', 'wasn', 'weren', 'wouldn']:
    stoplist.remove(w)

# Initialize NLTK function
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

# Initialize tokenizer
punc_tokenizer = RegexpTokenizer(r'\w+')

## Build Sentiment Lexicon
#https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
positiveWords = set(open(POSITIVE_NEGATIVE_WORDS_PATH + "positive-words.txt", encoding = "ISO-8859-1").read().split())
negativeWords = set(open(POSITIVE_NEGATIVE_WORDS_PATH + "negative-words.txt", encoding = "ISO-8859-1").read().split())

# intialize en_US dictionnary 
d = enchant.Dict("en_US")

# this dictionnary will contain the spelling correction collected from 3 dictionnary found on internet.
dico = {}

# process the firt dictionnary
dico1 = open(SPELLING_DICT_PATH + 'correct_spelling_1.txt', 'rb')
for word in dico1:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
dico1.close()

# process the second dictionnary
dico2 = open(SPELLING_DICT_PATH + 'correct_spelling_2.txt', 'rb')
for word in dico2:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico2.close()

# process the third dictionnary
dico3 = open(SPELLING_DICT_PATH + 'correct_spelling_2.txt', 'rb')
for word in dico3:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico3.close()

# slang words are words like 4u, brb ... we will correct them using slang dictionnary downloaded from internet
slang_dic_initial = open(SLANG_WORDS_DICT_PATH + 'slang_words.txt', 'rb')

# slang_dic will contain the slang words
slang_dic = {}

# process dictionnary 
for word in slang_dic_initial:
    word = word.decode('utf8')
    word = word.split()
    slang_dic[word[0]] = word[1]
slang_dic_initial.close()


def url_user(text):
    # Removal of URLs
    text = re.sub(r"<url>","",str(text))
    # Removal of user
    text = re.sub(r"<user>","",str(text))
    return text.strip()

def split_hashtag(text):
    '''
    This method removes # from hashtag and if it's composed of more than one word 
    it will split it into different words
    example: #machinelearning ---> machine learning
    '''
    
    t = []
    text = re.sub(r"#", " #", text)
    text_ = []
    for w in text.split():
        if w.startswith("#"):
            w = re.sub(r'#(\S+)', r' \1 ', w)
            t = wordninja.split(w)
            w = (" ".join(t)).strip()
            text_.append(w)
        else: text_.append(w)
    return (" ".join(text_)).strip()

def emphasize_sentiment_words(text):
    '''
    This method adds the word positive if the word processed is in the positive dictionnary
    or it adds the word negative if the word proceesed is in the negative dictionnary
    '''
    t = []
    for w in text.split():
        if w in positiveWords:
            t.append('positive ' + w)
        elif w in negativeWords:
            t.append('negative ' + w)
        else:
            t.append(w)
    return (" ".join(t)).strip()


def remove_number(text):
    '''
    This method removes numbers from the tweet
    '''
    new_text = []
    for word in text.split():
        try:
            word = re.sub('[,\.:%_\-\+\*\/\%\_]', '', word)
            float(word)
            new_text.append("")
        except:
            new_text.append(word)
    return " ".join(new_text)

def filter_small_words(text):
    '''
    this method removes words with length less than one.
    '''
    return " ".join([w for w in text.split() if len(w) >1 or not w.isalpha()])

def clean_punctuation(text):
    '''
    This method processes punctuation
    '''
    # Apostrophe
    text = re.sub(r"[^a-zA-Z]"," ",str(text))
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\.", " \. ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Removal of mentions
    text = re.sub("@[^\s]*", "", text)
    return text

def filter_punctuation(text):
    '''
    This method removes punctuation from the tweet
    '''
    return " ".join(punc_tokenizer.tokenize(text))


def slang_words(text):
    '''
    This method correct slang words present in the tweet using slang dictionnary
    '''
    text = text.split()
    for i in range(len(text)):
        if text[i] in slang_dic.keys():
            text[i] = slang_dic[text[i]]
    text = ' '.join(text)
    return text


def apostrophe(text):
    '''
    This methode transforms words with apostrophes at the end into two words
    '''
    # Apostrophe lookup
    text = re.sub(r"it\'s","it is",str(text))
    text = re.sub(r"i\'d","i would",str(text))
    text = re.sub(r"don\'t","do not",str(text))
    text = re.sub(r"he\'s","he is",str(text))
    text = re.sub(r"there\'s","there is",str(text))
    text = re.sub(r"that\'s","that is",str(text))
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"who\'s", "who is", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'s"," is",text)
    return text

def emoji_translation(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' positive ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positive ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' positive ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' positive ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' negative ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' negative ', text)
    return text


def remove_repetition(text):
    "llooooooovvvee becomes love"
    text=text.lower()
    text=text.split()
    for i in range(len(text)):
        if d.check(''.join(''.join(s)[:2] for _, s in itertools.groupby(text[i]))):
            text[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(text[i]))
        else:
            text[i]=''.join(''.join(s)[:1] for _, s in itertools.groupby(text[i]))
    text=' '.join(text)
    return text

def correct_spell(text):
    '''
    This method correct the spelling mistakes in the word
    '''
    text = text.split()
    for i in range(len(text)):
        if text[i] in dico.keys():
            text[i] = dico[text[i]]
    text = ' '.join(text)
    return text

def remove_stopwords(text):
    '''
    This method removes stop words from the tweet
    '''
    tokens = text.split()
    for word in tokens:
        if word in stoplist:
            tokens.remove(word)
    return ' '.join(tokens)

#  Lemmatization
def lemmatize(text):
    '''
    lemmatize words: plays, playing ,played --> play
    '''
    words = text.split()
    lemmatized = list()
    for word in words:
        try:
            lemmatized.append(lemma.lemmatize(word).lower())  
        except Exception:
             lemmatized.append(word)
    return " ".join(lemmatized)

# Stemming
def stemming(text):
    '''
    stemm words: Car, cars --> car
    '''
    x = [stemmer.stem(t) for t in text.split()]
    return " ".join(x)

# This function load data and process tweet using methods defined above
def load_data_and_labels(positive_data_file, negative_data_file,test_data_file, HASHTAG = True, EMPHASIZE = True, FILTER_PUNC=True, NUM =True, SMALL_WORDS = True , \
                        CLEAN_PUN = True, SLANG =True, APOSTROPHE = True, EMOJI = True, REPITITION = True, SPELL = True, \
                        STOPWORDS = True, LEMMATIZE = True, STEMMING = True):
    
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    test_data = list(open(test_data_file, "r", encoding='utf-8').readlines())
    test_data = [s.strip() for s in test_data]
    
    # Split by words
    print("Starting Data processing")
    x = positive_examples + negative_examples
    
    # remove url and user
    x = [url_user(sent) for sent in x]
    test_data = [url_user(sent) for sent in test_data]

    if HASHTAG:
        print("processing hashtag")
        x = [split_hashtag(sent) for sent in x]
        test_data = [split_hashtag(sent) for sent in test_data]
        
    if EMPHASIZE:
        print("Processing emphasize sentiment words")
        x = [emphasize_sentiment_words(sent) for sent in x]
        test_data = [emphasize_sentiment_words(sent) for sent in test_data]
        
    if EMOJI:
        print("Processing emojis")
        x = [emoji_translation(sent) for sent in x]
        test_data = [emoji_translation(sent) for sent in test_data]
        
    if FILTER_PUNC:
        print("Processing duplicated punctuation")
        x = [filter_punctuation(sent) for sent in x]
        test_data = [filter_punctuation(sent) for sent in test_data]
        
    if SMALL_WORDS:
        print("Processing small words")
        x = [filter_small_words(sent) for sent in x]
        test_data = [filter_small_words(sent) for sent in test_data]
        
    if SLANG:
        print("Processing slang words")
        x = [slang_words(sent) for sent in x]
        test_data = [slang_words(sent) for sent in test_data]
        
    if CLEAN_PUN:
        print("Cleaning punctuation")
        x = [clean_punctuation(sent) for sent in x]
        test_data = [clean_punctuation(sent) for sent in test_data]
        
    if APOSTROPHE:
        print("Processing apostrophes")
        x = [apostrophe(sent) for sent in x]
        test_data = [apostrophe(sent) for sent in test_data]
        
    if REPITITION:
        print("Processing repetition")
        x = [remove_repetition(sent) for sent in x]
        test_data = [remove_repetition(sent) for sent in test_data]
        
    if SPELL:
        print("Processing spelling mistakes")
        x = [correct_spell(sent) for sent in x]
        test_data = [correct_spell(sent) for sent in test_data]
        
    if NUM:
        print("Processing numbers")
        x = [remove_number(sent) for sent in x]
        test_data = [remove_number(sent) for sent in test_data]
        
    if STOPWORDS:
        print("Processing stop words")
        x = [remove_stopwords(sent) for sent in x]
        test_data = [remove_stopwords(sent) for sent in test_data]
        
    if LEMMATIZE:
        print("Lemmatizing")
        x = [lemmatize(sent) for sent in x]
        test_data = [lemmatize(sent) for sent in test_data]
        
    if STEMMING:
        print("Stemming")
        x = [stemming(sent) for sent in x]
        test_data = [stemming(sent) for sent in test_data]
    print("Returning X data and kaggle data")
    return [x, test_data]