import re, string
import numpy as np 
import multiprocessing

import jieba
jieba.enable_parallel(multiprocessing.cpu_count())
user_dict = "./data/langrensha.txt"
jieba.load_userdict(user_dict)

# ==============================================================================
# preprocessing
# ==============================================================================
def generate_seglist(sentence,stopwords):
    words = jieba.cut(sentence)
    final = []
    for word in words:
        if word not in stopwords:
            final.append(word)
    return final

def remove_punctuation(words):
    regex = re.compile('[%s]' % re.escape(string.punctuation+'、'))
    return regex.sub('', words)

# ==============================================================================
# generate word2vec of a sentence
# ==============================================================================
def generate_sentence_vec(words, model, num_features):
    ## Function to average all of the word vectors in a given words list
    
    ## Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    ## Index2word is a list that contains the names of the words in 
    ## the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    ## Loop over each word in the review and, if it is in the model's
    ## vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    ## Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec