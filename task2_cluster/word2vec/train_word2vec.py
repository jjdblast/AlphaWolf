# -*- coding: utf-8 -*-
# train_word2vec.py用于训练模型

import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

## function generate word2vec and save to model, vector
def generate_word2vec_from_file(input_corpus, output_model, output_vector):
    model = Word2Vec(LineSentence(input_corpus),
                     size=400,window=5,min_count=5,
                     workers=multiprocessing.cpu_count())
    model.save(output_model)
    model.wv.save_word2vec_format(output_vector,binary=false)
    
def generate_word2vec(df, output_model, output_vector):
    model = Word2Vec(df,
                     size=400,window=5,min_count=5,
                     workers=multiprocessing.cpu_count())
    model.save(output_model)
    model.wv.save_word2vec_format(output_vector,binary=False)
    return model
    

if __name__=='__main__':
    ## get logger from main program
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    ## output process
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))
    ## retrun exit when input inccorect
    if len(sys.argv) < 4:
        sys.exit(1)
    
    ## input corpus, output model location, output vector location
    inp,outp,outp2 = sys.argv[1:4]
    
    generate_word2vec_from_file(inp,outp,outp2)

