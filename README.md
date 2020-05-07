# Sentiment Analysis Twitter
This project was done in the scope of the course Machine Learning at EPFL The competition was hosted at www.crowdai.org.
We were given dataset of 2500000 tweets, half of it conatins positive labels and the other half contains negative labels.
For the comptetions we used a test dataset that contains 10000 entries.

# Necessary libraries
- numpy
- keras
- tensorflow
- sklearn
- matplotlib
- pickle
- enchant
- wordninja
- re
- glove
- os
- math

# Folders
- cleaned_data: This folder contains the cleaned data after being processed.
- dict_correct_spelling: This folder contains dictionnaries for correct spelling collected from internet
      - http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
      - http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
      - http://luululu.com/tweet/typo-corpus-r1.txt
- dict_slan_words: This folder contains a dictionnary with slang word corrected. Example: 4u --> for you. this dictionnary was constructed by us
- glove_embedding: this folder contains the glove weights and words embedding, also it contains the tweet-glove downloaded from stanford.
- positive_negative_words: This folder contains two dictionnairies, the first one contains positive words and the second one contains negative words.
- train_test_data: This folder contains the train and test data.

# Code and notebooks
- helper.py: This script contains methods to read and process the data
- process.py: This script calls specific methods of helper.py to clean the data and save the cleaned version
- paths.py: It contains only the paths used in our code so that we won't define them seperately.
- tfidfi_methods.py: This script will use the glove weighted matrix and average the vectors of the words in each tweet.
- kaggle_submission.py: This script create submission file from saved models.
- create_word_vectors.py: This script create a word embedding using either stanford pretrained files or by constructing our own glove.
- run_models: This script contains the models that we used to generate kaggle predicitons.
- run.py: This script run the project and call the above scripts and cerates the kaggle sumbimission.
- tf_idf_models.ipynb: This notebook contain models training on TF-IDF.  

# Run our project
To be able to run our project you need first to install the above librairies and then:
  - Download glove.twitter.27B.zip from
https://nlp.stanford.edu/projects/glove/?fbclid=IwAR1yRzuBFvrUYngB61tEOLXlYoqTaBjbnzJmxz4TQcSIfh4YFYZaXPIyYfA and extract it in 'glove_embedding' folder.
  - Download the train and test data from https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files
and extract them in 'train_test_data' folder
  - run the script run.py

# Useful links:
- https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-11-cnn-word2vec-41f5e28eda74
- https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
- https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# Collaborators
* Mouadh Hamdi: mouadh.hamdi@epfl.ch
* Mariem Belhaj Ali: mariem.belhajali@epfl.ch
* Nourchene Ben Romdhane: nourchene.benromdhane@epfl.ch
