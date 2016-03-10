import os
import json
import io
from stop_words import get_stop_words
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
from numpy import linalg as LA
from operator import itemgetter
import itertools
from scipy.special import expit
from scipy import sparse, io
from collections import OrderedDict
import csv
import pickle

path = "/data/shared/twitter/twitter_stream_data2"
en_stop_words = set(get_stop_words('en'))
es_stop_words = set(get_stop_words('spanish'))
hashtag_dict = {} # hashtag_dict={common_hahstag: {es: {id1, id2, ...}, en: {id1, id2, ...}}} # if len(hashtag_dict[hashtag])==2, we have found one commnon hashtag between two languages
tfidf_vectorizer = TfidfVectorizer(min_df = 1, sublinear_tf = True) 
enTweet_ID_dict = {} # {enTweetPrunedText : ID}
esTweet_ID_dict = {} # {esTweetPrunedText : ID}
enID_hashtag_dict = {} # {ID: {hashtag1,2,...} or 'notag'}
esID_hashtag_dict = {} # {ID: {hashtag1,2,...} or 'notag'}

def preprocessText(text):
	text = text.lower()
	text = re.sub('rt', '', text)   # remove "RT"
	text = re.sub('((https?://[^\s]+)|(http?://[^\s]+))', '', text)   # remove urls
	text = re.sub('@[^\s]+', '', text)   # remove usernames
	# text = re.sub(r'#([^\s]+)', '', text)   # remove hashtags
	text = re.sub(r'#', '', text)   # remove the '#' prefix of hashtags
	text = re.sub('\s+', ' ', text).strip(' ') # turn all whitespaces to a single space
	text = "".join([ch for ch in text if ch not in string.punctuation]) # remove punctuation
	return text

for filename in os.listdir(path):

	print "reading " + filename
	filepath = path + filename
	fd = open(filepath, 'r') # In python2, open() doesn't have encoding param. --> io.open(, encoding='gbk')

	for line in fd.read().rstrip().split("\n"):
		if line=="":
			continue
		stream_data = line.split("\t")
		jsonObj = json.loads(stream_data[1]) 
		lang = jsonObj["lang"]
		hashtags = jsonObj["entities"]["hashtags"] #a list of values, e.g: "entities":{"hashtags":[{"text":"LG","indices":[78,81]},{"text":"Smoothblink2Jumia","indices":[111,129]}],"urls":...
		id = str(jsonObj["id"])
		raw_text = jsonObj["text"]
		isRetweeted = jsonObj["retweeted"]
		if lang == "en":
			tmp_text = preprocessText(raw_text) # remove hashtag, username, urls
			pruned_text = ' '.join([word for word in tmp_text.split() if word not in en_stop_words]) # remove stopwords
			if pruned_text != '':
				enTweet_ID_dict[pruned_text] = id # don't consider many ids have same text, overwrite previous ids sharing same text, save the last id
				if hashtags == []:
					enID_hashtag_dict[id] = "notag"
				if hashtags != []: 
					enID_hashtag_dict[id] = set()
					for hashtagObj in hashtags:
						hashtag = hashtagObj["text"]
						enID_hashtag_dict[id].add(hashtag)
						if hashtag not in hashtag_dict:
							hashtag_dict[hashtag] = {}
						if lang not in hashtag_dict[hashtag]:
							hashtag_dict[hashtag][lang]= set()
						if not isRetweeted:
							hashtag_dict[hashtag][lang].add(id)
		if lang == "es":
			tmp_text = preprocessText(raw_text) # remove hashtag, username, urls
			pruned_text = ' '.join([word for word in tmp_text.split() if word not in es_stop_words])
			if pruned_text != '':
				esTweet_ID_dict[pruned_text] = id
				if hashtags == []:
					esID_hashtag_dict[id] = "notag"
				if hashtags != []: 
					esID_hashtag_dict[id] = set()
					for hashtagObj in hashtags:
						hashtag = hashtagObj["text"]
						esID_hashtag_dict[id].add(hashtag)
						if hashtag in hashtag_dict:
							if lang not in hashtag_dict[hashtag]:
								hashtag_dict[hashtag][lang]= set()
							if not isRetweeted:
								hashtag_dict[hashtag][lang].add(id)

print "The number of hashtags in hashtag_dict is " + str(len(hashtag_dict)) # 163696
print "The number of tweets in en_corpus is : " + str(len(enTweet_ID_dict)) # 1316682
print "The number of tweets in es_corpus is : " + str(len(esTweet_ID_dict)) # 392018

enToken_count_dict = {} # {token: occurrence}
esToken_count_dict = {} # {token: occurrence}

def token_count(text_list, count_dict):
	for text in text_list:
		for token in re.split('\s', text): # split with whitespace
			try:
				count_dict[token] += 1
			except KeyError:
				count_dict[token] = 1

token_count(enTweet_ID_dict.keys(), enToken_count_dict)
token_count(esTweet_ID_dict.keys(), esToken_count_dict)

print "The number of english tokens is " + str(len(enToken_count_dict)) # 621335
print "The number of spanish tokens is " + str(len(esToken_count_dict)) # 266603

freq_enTokens_set = set()
infreq_enTokens_set = set()
freq_esTokens_set = set()
infreq_esTokens_set = set()

freq_enTokens_threshhold = int(0.00008 * len(enToken_count_dict))
freq_esTokens_threshhold = int(0.0004 * len(esToken_count_dict)) 
infreq_enTokens_threshhold = 7 #10 #9
infreq_esTokens_threshhold = 5 #7 #6 

for key in enToken_count_dict:
	if enToken_count_dict[key] >= freq_enTokens_threshhold:
		freq_enTokens_set.add(key)
	if enToken_count_dict[key] <= infreq_enTokens_threshhold:
		infreq_enTokens_set.add(key)

for key in esToken_count_dict:
	if esToken_count_dict[key] >= freq_esTokens_threshhold:
		freq_esTokens_set.add(key)
	if esToken_count_dict[key] <= infreq_esTokens_threshhold:
		infreq_esTokens_set.add(key)


print "The number of frequent english tokens is " + str(len(freq_enTokens_set)) # 21171 with threshhold=0.00005
print "The number of infrequent english tokens is " + str(len(infreq_enTokens_set)) # 579136 less than 10 times occurrence
print "The number of frequent and infrequnet english tokens is " + str(len(freq_enTokens_set)+len(infreq_enTokens_set)) # 600307
print "The number of frequent spanish tokens is " + str(len(freq_esTokens_set)) # 11417 with threshhold=0.0001
print "The number of infrequent spanish tokens is " + str(len(infreq_esTokens_set)) # 238330 less than 7 times occurrence
print "The number of frequent and infrequnet spanish tokens is " + str(len(freq_esTokens_set)+len(infreq_esTokens_set)) # 249747


# return a new_tweet_ID_OrderedDict without duplicates tweets 
def removeFreqOrInfreTokens(tweet_ID_dict, freq_set, infreq_set): 
	new_tweet_ID_OrderedDict = OrderedDict()
	new_ID_tweet_dict = {}
	for key in tweet_ID_dict:
		new_key = ' '.join([token for token in key.split() if token not in freq_set and token not in infreq_set])
		if new_key != '':
			tweetID = tweet_ID_dict[key]
			new_tweet_ID_OrderedDict[new_key] = tweetID
			new_ID_tweet_dict[tweetID] = new_key
	return new_tweet_ID_OrderedDict, new_ID_tweet_dict


new_enTweet_ID_OrderedDict, new_enID_tweet_dict = removeFreqOrInfreTokens(enTweet_ID_dict, freq_enTokens_set, infreq_enTokens_set)
new_esTweet_ID_OrderedDict, new_esID_tweet_dict = removeFreqOrInfreTokens(esTweet_ID_dict, freq_esTokens_set, infreq_esTokens_set)
print "The new number of tweets in en_corpus after removeFreqOrInfreTokens is : " + str(len(new_enTweet_ID_OrderedDict)) # 72269 ~5.5% of original size 
print "The new number of tweets in es_corpus after removeFreqOrInfreTokens is : " + str(len(new_esTweet_ID_OrderedDict)) # 62786 ~16% of original size 

def mapTweetID2RowID(tweet_ID_OrderedDict):
	tweetID_rowID_dict = {}
	tweetIDs_list = tweet_ID_OrderedDict.values()
	for index, tweetID in enumerate(tweetIDs_list):
		tweetID_rowID_dict[tweetID] = index
	return tweetID_rowID_dict

enTweetID_rowID_dict = mapTweetID2RowID(new_enTweet_ID_OrderedDict)
esTweetID_rowID_dict = mapTweetID2RowID(new_esTweet_ID_OrderedDict)
print "The number of items in enTweetID_rowID_dict is " + str(len(enTweetID_rowID_dict)) # 72269 
print "The number of items in esTweetID_rowID_dict is " + str(len(esTweetID_rowID_dict)) # 62786 

def save_obj(obj, name):
    with open('/data/shared/twitter/output/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(esTweetID_rowID_dict, 'esTweetID2rowID5')
save_obj(enTweetID_rowID_dict, 'enTweetID2rowID5')
save_obj(enID_hashtag_dict, 'enID2hashtag5')
save_obj(esID_hashtag_dict, 'esID2hashtag5')

enTweet_ID_dict.clear()
esTweet_ID_dict.clear()

new_enToken_count_dict = {} # {token: occurrence}
new_esToken_count_dict = {} # {token: occurrence}
token_count(new_enTweet_ID_OrderedDict.keys(), new_enToken_count_dict)
token_count(new_esTweet_ID_OrderedDict.keys(), new_esToken_count_dict)
print "The number of english tokens after removeFreqOrInfreTokens is " + str(len(new_enToken_count_dict)) # 21028 
print "The number of spanish tokens after removeFreqOrInfreTokens is " + str(len(new_esToken_count_dict)) # 16856 

en_corpus_sparseMat = tfidf_vectorizer.fit_transform(new_enTweet_ID_OrderedDict.keys()) 
print "The english sparse tf-idf matrix dimension(# of rows is # of tweets, # of cols is # of features) is "
print en_corpus_sparseMat.shape # <72269, 19981>
es_corpus_sparseMat = tfidf_vectorizer.fit_transform(new_esTweet_ID_OrderedDict.keys()) # transform es_corpus into tf-idf valued sparse matrix
print "The spanish sparse tf-idf matrix dimension(# of rows is # of tweets, # of cols is # of features) is "
print es_corpus_sparseMat.shape # <62786, 16267>

io.mmwrite("/data/shared/twitter/output/en_corpus_sparseMat5.mtx", en_corpus_sparseMat)
io.mmwrite("/data/shared/twitter/output/es_corpus_sparseMat5.mtx", es_corpus_sparseMat)

esTweetID_set = set() # set of all es_tweetIDs in the hashtag_dict that appear in the new_esTweet_ID_OrderedDict

# build en_es+ dictionary: {en_id: {es_id1, es_id2, ...}}
tweetsPairGroup_pos = {}
cmnHashtag_count = 0
for hashtag in hashtag_dict:
	if len(hashtag) > 2 and len(hashtag_dict[hashtag])== 2:
		cmnHashtag_count += 1
		for es_id in hashtag_dict[hashtag]['es']:
			# build esTweetID_set:
			if es_id in new_esID_tweet_dict:
				esTweetID_set.add(es_id)
				for en_id in hashtag_dict[hashtag]['en']:
					if en_id in new_enID_tweet_dict:
						if en_id not in tweetsPairGroup_pos:
							tweetsPairGroup_pos[en_id] = set() # initialize
						tweetsPairGroup_pos[en_id].add(es_id)
					
new_esTweet_IDs = set(new_esTweet_ID_OrderedDict.values()) 
intersect_esID_set = new_esTweet_IDs.intersection(esTweetID_set)
print "The size of new_esTweet_ID_OrderedDict and new_esTweet_IDs is " + str(len(new_esTweet_ID_OrderedDict)) # 62786 
print "The size of esTweetID_set is " + str(len(esTweetID_set)) # 14706 
print "The size of intersect_esID_set is "+ str(len(intersect_esID_set)) # 6256 
print "The number of common hashtags with length>2 between english and spanish tweets is " + str(cmnHashtag_count) # 5880 
print "The number of en_ids in tweetsPairGroup_pos is " + str(len(tweetsPairGroup_pos)) # 43108 

new_enTweet_ID_OrderedDict.clear()
new_esTweet_ID_OrderedDict.clear()

# build data collection for train and test
en_esPos_esNeg_dict = {} # {(en_id1,es+_id1):es-_id1, (en_id1,es+_id2):es-_id2,...} 
for en_id in tweetsPairGroup_pos:
	esPos_ids = tweetsPairGroup_pos[en_id] # esPos_ids is a set
	esPos_len = len(esPos_ids)
	esNeg_ids = list(esTweetID_set - esPos_ids)
	if esPos_len < len(esNeg_ids):
		esNeg_ids_sample = random.sample(esNeg_ids, esPos_len)
		for i in range(esPos_len):
			en_esPos_esNeg_dict[tuple((en_id, list(esPos_ids)[i]))] = esNeg_ids_sample[i]
	else:
		for i in range(len(esNeg_ids)):
			en_esPos_esNeg_dict[tuple((en_id, list(esPos_ids)[i]))] = esNeg_ids[i]

print "The number of items in en_esPos_esNeg_dict is " + str(len(en_esPos_esNeg_dict)) # 594301 

tweetsPairGroup_pos.clear()

keys_list = en_esPos_esNeg_dict.keys() # List of keys
random.shuffle(keys_list) # Shuffle the list of keys in place

data_count = 0
with open('/data/shared/twitter/output/207output5.csv', 'wb') as csvfile:
	fieldnames = ['en_tweet_ID', 'en_tweet_text', 'en_tweet_hashtag', 'es_pos_tweet_ID', 'es_pos_tweet_text', 'es_pos_tweet_hashtag', 'es_neg_tweet_ID', 'es_neg_tweet_text', 'es_neg_tweet_hashtag', ]
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for k in keys_list:
		en_tweet_ID = k[0]
		es_pos_tweet_ID = k[1]
		es_neg_tweet_ID = en_esPos_esNeg_dict[k]
		if en_tweet_ID in enTweetID_rowID_dict and es_pos_tweet_ID in esTweetID_rowID_dict and es_neg_tweet_ID in esTweetID_rowID_dict:
			data_count += 1
			en_tweet_text = new_enID_tweet_dict[en_tweet_ID]
			es_pos_tweet_text = new_esID_tweet_dict[es_pos_tweet_ID]
			es_neg_tweet_text = new_esID_tweet_dict[es_neg_tweet_ID]
			en_tweet_hashtag = enID_hashtag_dict[en_tweet_ID]
			es_pos_tweet_hashtag = esID_hashtag_dict[es_pos_tweet_ID]
			cmn_hashtag = list(en_tweet_hashtag & es_pos_tweet_hashtag)[0]
			es_neg_tweet_hashtag = list(esID_hashtag_dict[es_neg_tweet_ID])[0]
			writer.writerow({'en_tweet_ID': unicode(en_tweet_ID).encode("utf-8"), 'en_tweet_text': unicode(en_tweet_text).encode("utf-8"), 'en_tweet_hashtag': unicode(cmn_hashtag).encode("utf-8"), 'es_pos_tweet_ID': unicode(es_pos_tweet_ID).encode("utf-8"), 'es_pos_tweet_text': unicode(es_pos_tweet_text).encode("utf-8"), 'es_pos_tweet_hashtag': unicode(cmn_hashtag).encode("utf-8"), 'es_neg_tweet_ID': unicode(es_neg_tweet_ID).encode("utf-8"), 'es_neg_tweet_text': unicode(es_neg_tweet_text).encode("utf-8"), 'es_neg_tweet_hashtag': unicode(es_neg_tweet_hashtag).encode("utf-8")})
		else:
			continue

print "The number of rows in csv should be", data_count
