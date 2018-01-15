# coding=utf-8

#情感分析
import os
os.chdir('D:\\MySubject\\project2')

import numpy as np 
import pandas as pd
from pandas import Series, DataFrame
import jieba
jieba.load_userdict('data/diction.txt')
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from scipy.sparse import csr_matrix
import sys


def word_map(text, worddict):
	'''
	map some word in text to another word, map dict is worddict
	'''
	for k in worddict:
		text = text.replace(k, worddict[k])
	return text



def text_preprocess(text):
	'''
	chinese text preprocess: jiebacut, filter special character
	'''
	wordlist = list(jieba.cut(text)) #为unicode
	text_cuted = ' '.join(wordlist)
	r = u'[a-zA-Z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
	text_clean = re.sub(r,'',text_cuted)
	return text_clean


def get_keyword(content, label, thredp=0.0, thredn=0.0):
	'''
	find label 0 contents' key words
	key word is word whose prob> thredp & count> thredn
	Parameters:
		content: text list
		label: float list, label[i] = label(content[i])
		thredp: float
		thredn: int
	return:
		key words dict: {word:(prob, wordcount),......}, prob = P(label(content)!=0| word in content)
	'''
	n = len(content)
	word_count = {}
	for i in range(n):
		content_ = content[i]
		label_ = 0 if label[i]==0 else 1
		wordlist_ = set(content_.split())
		for word_ in wordlist_:
			if word_ not in word_count:
				word_count[word_] = [0.0,0.0]
			word_count[word_][label_]+=1
	word_prob = {k: (word_count[k][1]/sum(word_count[k]), sum(word_count[k])) for k in word_count \
	if (word_count[k][1]/sum(word_count[k])>thredp) & (sum(word_count[k])>thredn)}
	return word_prob


class BiunigramVecer():
	"""generate binary unigram vector for corpus"""
	def fit_transform(self,corpus):
		vocabulary = {}
		nrow = len(corpus)
		nword = 0
		row_indlist = []
		col_indlist = []
		for row_ind in range(nrow):
			corpus_ = corpus[row_ind]
			wordset_ = set(corpus_.split())
			for word_ in wordset_:
				if word_ not in vocabulary:
					vocabulary[word_] = nword
					nword+=1
				col_ind = vocabulary[word_]
				col_indlist.append(col_ind)
				row_indlist.append(row_ind)
		ndata = len(col_indlist)
		biunigram_mtri = csr_matrix(([1.0]*ndata,(row_indlist, col_indlist)))
		self.vocabulary_ = vocabulary
		return biunigram_mtri

	def transform(self,corpus):
		vocabulary = self.vocabulary_
		nrow = len(corpus)
		ncol = len(vocabulary)
		row_indlist = []
		col_indlist = []
		for row_ind in range(nrow):
			corpus_ = corpus[row_ind]
			wordset_ = set(corpus_.split())
			for word_ in wordset_:
				if word_ not in vocabulary:
					continue
				col_ind = vocabulary[word_]
				row_indlist.append(row_ind)
				col_indlist.append(col_ind)
		ndata = len(row_indlist)
		biunigram_mtri = csr_matrix(([1.0]*ndata,(row_indlist, col_indlist)),shape=(nrow, ncol))
		return biunigram_mtri

def filter_onkeyword(text, keyword):
	'''
	judge if the text has any keyword
	Parameters:
		text: str, clean text
		keyword: dict or set
	return:
		bool
	'''	
	wordlist = text.split()
	ifkey = [w in keyword for w in wordlist]
	return any(ifkey)




worddict = {u'KS':u'1号',u'毛毛':u'2号',u'饮料':u'3号',u'李斯':u'4号',u'JY':u'5号',u'王师傅':u'6号'\
,u'李锦':u'7号',u'大宝':u'8号',u'囚徒':u'9号',u'小苍':u'10号',u'少帮主':u'11号',u'桃子':u'12号'}

data = pd.read_csv('data/data.csv')
data['emotion'].fillna(0,inplace=True)
data['content'] = data['content'].map(lambda x: x.decode('utf8'))
data['content_clean'] = data['content'].map(lambda x: word_map(x, worddict)).map(text_preprocess)
data['label'] = 0
data.loc[data['emotion']<0, 'label']=-1
data.loc[data['emotion']>0, 'label']=1



N = data.shape[0]
trainN = 2*N/3
traindata = data.sample(trainN, random_state=0)
testdata = data.drop(traindata.index.tolist())
train_label = np.array(traindata['label'].tolist())
test_label = np.array(testdata['label'].tolist())

keyword = get_keyword(traindata['content_clean'].tolist(), traindata['label'].tolist(), thredp=0.4, thredn=0)
a = traindata['content_clean'].map(lambda x: filter_onkeyword(x, keyword))
b = traindata[(1-a).astype(bool)]
b[b['emotion']!=0]



#SVC方法
#经实验，自写的binary unigram最好, sklearn的binary unigram(count vector)次之, tfidf 最差
#自写的binary unigram函数，无过滤任何词
biunigramvecer = BiunigramVecer()
train_biu_mtri = biunigramvecer.fit_transform(traindata['content_clean'].tolist())
test_biu_mtri = biunigramvecer.transform(testdata['content_clean'].tolist())
biumodel = svm.SVC(kernel = 'linear', probability = True)
biumodel.fit(train_biu_mtri, traindata['label'].tolist())
train_biupre_label = biumodel.predict(train_biu_mtri)
test_biupre_label = biumodel.predict(test_biu_mtri)

#sklearn的binary unigram，有过滤单词
vectorizer = CountVectorizer(binary=True)
train_vc_mtri =vectorizer.fit_transform(traindata['content_clean'].tolist())
test_vc_mtri = vectorizer.transform(testdata['content_clean'].tolist())
vcmodel = svm.SVC(kernel = 'linear', probability = True)
vcmodel.fit(train_vc_mtri, traindata['label'].tolist())
train_vcpre_label = vcmodel.predict(train_vc_mtri)
test_vcpre_label = vcmodel.predict(test_vc_mtri)

#sklearn的tfidf，有过滤单词
transformer = TfidfTransformer()
train_tfidf_mtri = transformer.fit_transform(train_vc_mtri)
test_tfidf_mtri = transformer.transform(test_vc_mtri)
tfidfmodel = svm.SVC(kernel = 'linear', probability = True)
tfidfmodel.fit(train_tfidf_mtri, traindata['label'].tolist())
train_tfidfpre_label = tfidfmodel.predict(train_tfidf_mtri) 
test_tfidfpre_label = tfidfmodel.predict(test_tfidf_mtri)



