import os
os.chdir('D:\\MySubject\\project2')

import numpy as np 
import pandas as pd
from pandas import Series, DataFrame
import re
import sys
import urllib
from sklearn.feature_extraction import DictVectorizer
import random
from sklearn import tree
from sklearn.externals.six import StringIO



def word_map(text, worddict):
	'''
	map some word in text to another word, map dict is worddict
	'''
	for k in worddict:
		text = text.replace(k, worddict[k])
	return text


def srltool(text):
	'''
	Semantic Role Labeling(SRL) of text, which is split by '。' to many sentences
	'''
	text = re.sub(u'[。；;？?！!…]',',',text)
	url_get_base = "http://api.ltp-cloud.com/analysis/?"
	args = {
		'api_key' : 'r1l17361j5EXeAKQVLgFISRYyu0xvAxhfdpj7eMc',
		'text' : text.encode('utf8'),
		'pattern' : 'srl',
		'format' : 'plain'
	}
	while(1):
		result = urllib.urlopen(url_get_base+urllib.urlencode(args)) # POST method
		srllist = result.readlines()
		srls_ = '\n'.join(srllist)
		if ('error' not in srls_) & ('ERROR' not in srls_):
			break
	srlresult = []
	for srltext_ in srllist:
		srlresult.append(srltext_.strip().decode('utf8'))
	return srlresult



def wstool(text):
	'''
	Word Segmentation(WS) of text, which is split by '。' to many sentences
	'''
	url_get_base = "http://api.ltp-cloud.com/analysis/?"
	args = {
		'api_key' : 'r1l17361j5EXeAKQVLgFISRYyu0xvAxhfdpj7eMc',
		'text' : text,
		'pattern' : 'ws',
		'format' : 'plain'
	}
	result = urllib.urlopen(url_get_base+urllib.urlencode(args)) # POST method
	wslist = result.readlines()
	wsresult = []
	for wstext_ in wslist:
		wstext_ = wstext_.strip()
		wsresult.append(wstext_.decode('utf8'))
	return wsresult


def ws(contentlist):
	'''
	Word Segmentation(WS) of every short content(unicode) in contentlist
	Return:
		list: [wsplain,...], wsplain = WS(contentlist[k])
	'''
	clean_contentlist = [x.replace(u'。', u'，').encode('utf8') for x in contentlist]
	
	n = len(clean_contentlist)
	candi = ''
	wsresult = []
	for k in range(n):
		content_ = clean_contentlist[k]
		if sys.getsizeof(candi+content_+'。')>=1024:
			result_ = wstool(candi)
			wsresult.extend(result_)

			candi = ''
		candi = candi+content_+'。'

	result_ = wstool(candi)
	wsresult.extend(result_)
	return wsresult


def object_strdeal(plain):
	'''
	example: '我认为7号是坏人'==>'我.7'，只保留 我、他、她和数字代号，中间以.分割
	'''
	new_plain = re.sub(u'[^0-9我你他她]',' ',plain)
	w_object = '.'.join(new_plain.split())
	return w_object


def clean_word(w):
	if '[' in w:
		pos0 = w.find('[')
		w = w[pos0+1:]
	if ']' in w:
		pos1 = w.find(']')
		w = w[:pos1]
	return w

def get_actchain(srlinfo_list):
	srl_word_list =srlinfo_list[0].split()
	nword = len(srl_word_list)
	word_list = [clean_word(w) for w in srl_word_list]
	label_list = [0]*nword

	for srlinfo in srlinfo_list:
		srl_word_list = srlinfo.split()
		for i in range(nword):
			w = srl_word_list[i]

			if '[' in w:
				pos0 = i
			if ']' in w:
				pos1 = i
				n_pos0 = w.find(']')
				name_ = w[n_pos0+1:]
				if name_ in ['v','p']:
					for j in range(pos0, pos1+1):
						label_list[j] = 1
#				elif name_ in ['TMP','LOC','DIS']:
#					for j in range(pos0, pos1+1):
#						label_list[j] = -1
			cleanw = word_list[i]
			if cleanw in emotion_words:
				label_list[i] = 1
			if cleanw in neg_words:
				label_list[i] = 1

	act_chain = []
	candi_ = ''
	for i in range(nword):
		if label_list[i]==-1:
			continue
		if label_list[i]==0:
			candi_ = candi_+' '+word_list[i]
			continue

		if candi_!='':
			object_ = object_strdeal(candi_)
			if object_!='':
				act_chain.append(object_)
			candi_ = ''
		act_chain.append(word_list[i])

	if candi_!='':
			act_chain.append(object_strdeal(candi_))
			candi_ = ''

	return ' '.join(act_chain),' '.join(word_list)

def find_last_index(list_, a):
	if a not in list_:
		return -1
	n = len(list_)
	list_reverse = [list_[n-1-i] for i in range(n)]
	index_ = list_reverse.index(a)
	return n-1-index_

def find_index(list_, a):
	if a not in list_:
		return -1
	return list_.index(a)

def match_pattern(pattern, word_list):
	'''
	example: 
	pattern =['object','v','center','v','object']
		['object','center','object']
		['object','v','object','v','center']
	['object','v','object','v','center','v','object']
	'''
	pattern_center_pos = pattern.index('center')
	word_pattern = ['']*len(pattern)

	word_label = []
	word_list_ = []
	prefix = ''
	for w in word_list:
		if w in neg_words:
			prefix = u'不'
			continue

		word_list_.append(prefix+w)
		prefix = ''
		if w in emotion_words:
			word_label.append('center')
		elif any([x in w for x in object_words+[u'我']]):
			word_label.append('object')
		else:
			word_label.append('v')
	word_list = word_list_

	center_pos = find_last_index(word_label, 'center')
	word_pattern[pattern_center_pos] = word_list[center_pos]

	if 'object' in pattern[pattern_center_pos:]:
		pattern_object_pos = find_index(pattern[pattern_center_pos:],'object')+pattern_center_pos
		object_pos = find_index(word_label[center_pos:],'object')+center_pos
		if object_pos>center_pos:
			word_pattern[pattern_object_pos] = word_list[object_pos]

	if 'v' in pattern[pattern_center_pos:]:
		pattern_v_pos = pattern_center_pos+1
		if object_pos>center_pos:
			v_pos = find_index(word_label[center_pos:object_pos],'v')+center_pos
		else:
			v_pos = find_index(word_label[center_pos:],'v')+center_pos
		if v_pos>center_pos:
			word_pattern[pattern_v_pos] = word_list[v_pos]

	if 'object' in pattern[:pattern_center_pos]:
		pattern_object_pos = find_last_index(pattern[:pattern_center_pos], 'object')
		object_pos = find_last_index(word_label[:center_pos],'object')
		if object_pos!=-1:
			word_pattern[pattern_object_pos] = word_list[object_pos]

	if 'v' in pattern[:pattern_center_pos]:
		pattern_v_pos = pattern_center_pos-1
		if object_pos!=-1:
			v_pos= find_last_index(word_label[object_pos: center_pos],'v')+object_pos
		else:
			v_pos= find_last_index(word_label[:center_pos],'v')
		if v_pos>object_pos:
			word_pattern[pattern_v_pos] = word_list[v_pos]

	if (object_pos!=-1):
		if 'object' in pattern[:pattern_object_pos]:
			pattern_object2_pos = 0
			object2_pos = find_last_index(word_label[:object_pos],'object')
			if object2_pos!=-1:
				word_pattern[pattern_object2_pos] = word_list[object2_pos]

		if 'v' in pattern[:pattern_object_pos]:
			pattern_v2_pos = 1
			if object2_pos!=-1:
				v2_pos= find_last_index(word_label[object2_pos: object_pos],'v')+object2_pos
			else:
				v2_pos= find_last_index(word_label[:object_pos],'v')
			if v2_pos>object2_pos:
				word_pattern[pattern_v2_pos] = word_list[v2_pos]

	return word_pattern








worddict = {u'KS':u'1号',u'毛毛':u'2号',u'饮料':u'3号',u'李斯':u'4号',u'JY':u'5号',u'王师傅':u'6号'\
,u'李锦':u'7号',u'大宝':u'8号',u'囚徒':u'9号',u'小苍':u'10号',u'少帮主':u'11号',u'桃子':u'12号',u'苍姐':u'10号'}


with open('words1') as f:
	lines = f.readlines()
words1 = [x.strip().decode('utf8') for x in lines if x.strip()!='']
with open('words2') as f:
	lines = f.readlines()
words2 = [x.strip().decode('utf8') for x in lines if x.strip()!='']
with open('words3') as f:
	lines = f.readlines()
words3 = [x.strip().decode('utf8') for x in lines if x.strip()!='']
with open('words4') as f:
	lines = f.readlines()
words4 = [x.strip().decode('utf8') for x in lines if x.strip()!='']
with open('words5') as f:
	lines = f.readlines()
words5 = [x.strip().decode('utf8') for x in lines if x.strip()!='']

emotion_words = list(set(words1+words2+words3+words4+words5))
words1set = set(words1)
words2set = set(words2)
words3set = set(words3)
words4set = set(words4)
words5set = set(words5)

neg_words = set([u'不',u'没',u'没有',u'不要'])
object_words = [str(i) for i in range(1, 13)]+[u'你',u'他',u'她']
ques_words = [u'吗',u'为什么']
if_words = [u'如果',u'假如',u'假设']


#data clean
#data = pd.read_csv('data/data_clean.csv', encoding = 'utf8')
def deal_emotion(s):
	if type(s)==float:
		return s
	s = s.replace(u'；',' ')
	s = s.replace(u';',' ')
	s = s.replace(u'’','')
	s = s.replace(u'‘','')
	slist = s.split()
	snumlist = [int(i) for i in slist]
	min_ = min(snumlist)
	max_ = max(snumlist)
	if min_+max_>0:
		return max_
	else:
		return min_

data = pd.read_csv('data/data.csv', encoding = 'utf8')
emotion_list = data['emotion'].map(deal_emotion)
data['emotion'] = emotion_list
data['emotion'].fillna(0,inplace=True)
data = data.loc[data['content'].notnull(),:]
data['content'] = data['content'].map(lambda x: word_map(x.replace(';','.'), worddict))
for i in data.index:
	content_ = data.loc[i,'content']
	order_ = data.loc[i,'order']
	data.loc[i,'content'] = content_

data['object'] = data['object'].map(lambda x: word_map(x, worddict) if type(x)!=float else '')
data['label'] = 0
data.loc[data['emotion']<0, 'label']=-1
data.loc[data['emotion']>0, 'label']=1
data.to_csv('data/data_clean.csv',index = False, encoding = 'utf8')

data = data.loc[:,['content','label']]

#data filter
data_srl = data.loc[data['content'].map(lambda x: any([i in x for i in emotion_words])),:].copy()
#wsresult = ws(data_srl_['content'].tolist())
#data_srl_['ws_content'] = wsresult

def havekeyword(text, keyword):
	textlist = text.split()
	return any([x in keyword for x in textlist])



#srlresult_list= map(srltool ,data_srl['content'].tolist())

srlresult_list = []
act_chain_list = []
ws_content_list = []


for i in data_srl.index.tolist():
	content_ = data_srl.loc[i, 'content']
	srlresult_ = srltool(content_)
	if srlresult_==[]:
		srlresult_ = ws([data_srl['content'].loc[i]])
	act_chain, ws_content  = get_actchain(srlresult_)

	act_chain_list.append(act_chain)
	ws_content_list.append(ws_content)
	srlresult_list.append(srlresult_)

data_srl['act_chain'] = act_chain_list
data_srl['ws_content'] = ws_content_list

'''
act_chain_list = []
ws_content_list = []
for i in range(len(srlresult_list)):
	srlresult_ = srlresult_list[i]
	act_chain, ws_content  = get_actchain(srlresult_)

	act_chain_list.append(act_chain)
	ws_content_list.append(ws_content)
data_srl['act_chain'] = act_chain_list
data_srl['ws_content'] = ws_content_list
'''

data_srl_filter = data_srl[data_srl['ws_content'].map(lambda x: havekeyword(x, set(emotion_words)))].copy()


act_chain_clean_list = []
for i in range(data_srl_filter.shape[0]):
	act_chain = data_srl_filter['act_chain'].iloc[i]
	r = match_pattern(['object','v','object','v','center','v','object'],act_chain.split())
	act_chain_clean = ' '.join(r)
	act_chain_clean_list.append(act_chain_clean)

data_srl_filter['act_chain_clean'] = act_chain_clean_list



data_srl.to_csv('data_srl.csv', encoding = 'utf8')
data_srl_filter.to_csv('data_srl_filter.csv', encoding = 'utf8')

#data_srl = pd.read_csv('data_srl.csv', index_col = 'index', encoding = 'utf8')
#data_srl_filter = pd.read_csv('data_srl_filter.csv', index_col = 'index', encoding = 'utf8')

#同类词映射
word_oldword = {u'think':[u'觉得',u'认为',u'认',u'感觉',u'说',u'确认',u'看',u'想'],\
u'not think':[u'不觉得',u'不认为',u'不认'],\
u'is':[u'是',u'像',u'有',u'开',u'就是',u'产'],\
u'isnt':[u'不是',u'不像'],\
u'not known':[u'不知道',u'不希望'],\
u'':[u'']}
oldword_word = {}
for word in word_oldword:
	for oldword in word_oldword[word]:
		oldword_word[oldword] = word

def clean_object(s):
	if s=='':
		return s
	obj_list = list(set(s.split('.')))
	if u'我' in obj_list:
		obj_list.remove(u'我')
		clean_obj = u'me'
	if u'你' in obj_list:
		obj_list.remove(u'你')
		clean_obj = u'you'
	if len(obj_list)>0:
		clean_obj = u'he'
	return clean_obj

def clean_v(s):
	if s not in oldword_word:
		return 'unimp'
	return oldword_word[s]

def clean_center(s):
	if s in words1set:
		return 'pos1'
	if s in words2set:
		return 'neg1'
	if s in words3set:
		return 'pos2'
	if s in words4set:
		return 'neg2'
	return 'pos3'

pattern = data_srl_filter.copy()

feature_list = []
dummyY = []
for i in pattern.index:
	act_chain_clean = pattern.loc[i, 'act_chain_clean']
	ws_content = pattern.loc[i,'ws_content'].split()
	wlist = act_chain_clean.split(' ')
	feature_ = {}
	if any([x in ws_content for x in if_words]):
		feature_['if'] = 'if'
	else:
		feature_['if'] = 'real'

	if any([x in ws_content for x in ques_words]):
		feature_['query'] = 'query'
	else:
		feature_['query'] = 'state'

	feature_['object0'] = clean_object(wlist[0])
	feature_['v0'] = clean_v(wlist[1])
	feature_['object1'] = clean_object(wlist[2])
	feature_['v1'] = clean_v(wlist[3])
	feature_['center'] = clean_center(wlist[4])
	feature_['v2'] = clean_v(wlist[5])
	feature_['object2'] = clean_object(wlist[6])
	feature_list.append(feature_)

	label_ = pattern.loc[i,'label']
	dummyY.append(label_)


vec = DictVectorizer()
dummyX = vec.fit_transform(feature_list).toarray()
dummyY = np.array(dummyY)


random.seed(3)
N = dummyX.shape[0]
trainN = N/3*2
train_index = random.sample(range(N), trainN)
trainX = dummyX[train_index,:]
trainY = dummyY[train_index]

test_index = [x for x in range(N) if x not in train_index]
testX = dummyX[test_index,:]
testY = dummyY[test_index]

clf = tree.DecisionTreeClassifier(random_state = 3, min_samples_leaf=5, criterion = 'entropy')
clf = clf.fit(trainX, trainY)

predict_trainY = clf.predict(trainX)
predict_testY = clf.predict(testX)


feature_list_ = [' '.join(map(lambda x: x if x!='' else '.',[x['if'],x['query'],x['object0'],x['v0'],x['object1'], x['v1'],x['center'],x['v2'],x['object2']])) for x in feature_list]
pattern['feature'] = feature_list_
pattern_train = pattern[['content','label','act_chain','ws_content','act_chain_clean','feature']].iloc[train_index].copy()
pattern_test = pattern[['content','label','act_chain','ws_content','act_chain_clean','feature']].iloc[test_index].copy()
pattern_train['Y'] = trainY
pattern_train['pred_Y'] = predict_trainY
pattern_test['Y'] = testY
pattern_test['pred_Y'] =  predict_testY

pd.crosstab(trainY, predict_trainY)
pd.crosstab(testY, predict_testY)


a = pattern[['feature','label']].groupby('feature').count()
a.sort_values('label',ascending = False, inplace = True)
big_featurelist = a.index[a['label']>=5].tolist()


pattern_b = pattern[pattern['feature'].map(lambda x: x in big_featurelist)].copy()
feature_list_b = []
dummyY_b = []
for i in pattern_b.index:
	act_chain_clean = pattern_b.loc[i, 'act_chain_clean']
	ws_content = pattern_b.loc[i,'ws_content'].split()
	wlist = act_chain_clean.split(' ')
	feature_ = {}
	if any([x in ws_content for x in if_words]):
		feature_['if'] = 'if'
	else:
		feature_['if'] = 'real'

	if any([x in ws_content for x in ques_words]):
		feature_['query'] = 'query'
	else:
		feature_['query'] = 'state'

	feature_['object0'] = clean_object(wlist[0])
	feature_['v0'] = clean_v(wlist[1])
	feature_['object1'] = clean_object(wlist[2])
	feature_['v1'] = clean_v(wlist[3])
	feature_['center'] = clean_center(wlist[4])
	feature_['v2'] = clean_v(wlist[5])
	feature_['object2'] = clean_object(wlist[6])
	feature_list_b.append(feature_)

	label_ = pattern_b.loc[i,'label']
	dummyY_b.append(label_)


vec = DictVectorizer()
dummyX_b = vec.fit_transform(feature_list_b).toarray()
dummyY_b = np.array(dummyY_b)


random.seed(3)
N = dummyX_b.shape[0]
trainN = N/3*2
train_index = random.sample(range(N), trainN)
trainX_b = dummyX_b[train_index,:]
trainY_b = dummyY_b[train_index]

test_index = [x for x in range(N) if x not in train_index]
testX_b = dummyX_b[test_index,:]
testY_b = dummyY_b[test_index]

clf = tree.DecisionTreeClassifier(random_state = 3, min_samples_leaf=1, criterion = 'entropy')
clf = clf.fit(trainX_b, trainY_b)

predict_trainY_b = clf.predict(trainX_b)
predict_testY_b = clf.predict(testX_b)


feature_list_b_ = [' '.join(map(lambda x: x if x!='' else '.',[x['if'],x['query'],x['object0'],x['v0'],x['object1'], x['v1'],x['center'],x['v2'],x['object2']])) for x in feature_list_b]
pattern_b['feature'] = feature_list_b_
pattern_b_train = pattern_b[['content','label','act_chain','ws_content','act_chain_clean','feature']].iloc[train_index].copy()
pattern_b_test = pattern_b[['content','label','act_chain','ws_content','act_chain_clean','feature']].iloc[test_index].copy()
pattern_b_train['Y'] = trainY_b
pattern_b_train['pred_Y'] = predict_trainY_b
pattern_b_test['Y'] = testY_b
pattern_b_test['pred_Y'] =  predict_testY_b

pd.crosstab(trainY_b, predict_trainY_b)
pd.crosstab(testY_b, predict_testY_b)


#Visulize model
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=vec.get_feature_names(),\
	filled=True, rounded=True, special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf('dectree_sample.pdf',encoding = 'utf8')


clf = tree.DecisionTreeClassifier(random_state = 3, min_samples_leaf=1, criterion = 'entropy')
clf = clf.fit(dummyX, dummyY)
predict_Y = clf.predict_proba(dummyX)
pattern['label=-1'] = predict_Y[:,0]
pattern['label=0'] = predict_Y[:,1]
pattern['label=1'] = predict_Y[:,2]

data['label=-1'] = 0.0
data['label=0'] = 1.0
data['label=1'] = 0.0
data.loc[pattern.index, 'label=-1'] = pattern['label=-1']
data.loc[pattern.index, 'label=0'] = pattern['label=0']
data.loc[pattern.index, 'label=1'] = pattern['label=1']

pattern[['content','label=-1','label=0','label=1']].to_csv('emotion_result_sub.csv', index = False, encoding = 'utf8')
data[['content','label=-1','label=0','label=1']].to_csv('emotion_result.csv', index = False, encoding = 'utf8')