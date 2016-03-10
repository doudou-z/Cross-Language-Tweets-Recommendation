import os
import json
import io
import random
import numpy as np
from numpy import linalg as LA
from operator import itemgetter
import itertools
from scipy.special import expit
from scipy import sparse, io
from scipy.sparse import csr_matrix
import csv
import pickle
import matplotlib.pyplot as plt

k = 100    # dimension of latent mapping space, ranging from 100 ~ 500
alpha = 0.05    # learning rate, later use adaptive learning rate

new_en_corpus_sparseMat = io.mmread("/data/shared/twitter/output/en_corpus_sparseMat4.mtx") #coo format; 'coo_matrix' object does not support indexing
new_es_corpus_sparseMat = io.mmread("/data/shared/twitter/output/es_corpus_sparseMat4.mtx") #coo format

en_corpus_sparseMat = csr_matrix(new_en_corpus_sparseMat) # convert coo format to csr format, which supports indexing
es_corpus_sparseMat = csr_matrix(new_es_corpus_sparseMat)

# print "en_corpus_sparseMat's type: ", type(en_corpus_sparseMat) #<class 'scipy.sparse.csr.csr_matrix'>
# print "en_corpus_sparseMat's shape: ", en_corpus_sparseMat.shape #

def load_obj(name):
    with open('/data/shared/twitter/output/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

enTweetID_rowID_dict = load_obj('enTweetID2rowID4')
esTweetID_rowID_dict = load_obj('esTweetID2rowID4')
# enID_hashtag_dict = load_obj('enID2hashtag5')
# esID_hashtag_dict = load_obj('esID2hashtag5')


Wnew = np.random.randn(en_corpus_sparseMat.shape[1], 100)
Qnew = np.random.randn(es_corpus_sparseMat.shape[1], 100)
Wold = Wnew
Qold = Qnew
Q_trans = np.transpose(Qold)   # k x Vs
is1stRun = True
iter_num = 0
iter_num2 = 0
isConverged = False
used_data = set()   # track the data used in iter_numations
H_list = []
L_list = []


with open('/data/shared/twitter/output/207output4.csv', 'rb') as csvfile:
	reader = csv.DictReader(csvfile)
	reader_list = list(reader)
	reader_list_len = len(reader_list)
	print "reader length: ", reader_list_len #317940 for 5 #27270 for 4
	train_size = int(0.8*len(reader_list))
	while True:
		position = random.randrange(0, train_size)
		rand_row = reader_list[position]
		used_data.add(position)
		en_id = rand_row['en_tweet_ID']
		esPos_id = rand_row['es_pos_tweet_ID']
		esNeg_id = rand_row['es_neg_tweet_ID']
		en_vec = en_corpus_sparseMat[enTweetID_rowID_dict[en_id]]   # 1 x Ve 
		enW = en_vec.dot(Wold)   # 1 x Ve, Ve x k -> 1 x k; A.dot(B) for sparse matrix and ndarray multiplication, while np.dot(A, B) only for ndarrays
		esPos_vec = es_corpus_sparseMat[esTweetID_rowID_dict[esPos_id]]   # 1 x Vs
		esNeg_vec = es_corpus_sparseMat[esTweetID_rowID_dict[esNeg_id]]   # 1 x Vs
		esDiff = esPos_vec - esNeg_vec   # 1 x Vs
		esDiff_T = esDiff.transpose()
		esDiff_Qold = esDiff * Qold # 1 x Vs, Vs x k -> 1 x k
		en_vec_T = en_vec.transpose()
		QTrans_esDiffTrans = esDiff_Qold.transpose()   # (1 x k)transpose -> k x 1
		H = np.dot(enW, QTrans_esDiffTrans)[0][0]   # H is a scalar Note: enW*QTrans_esDiffTrans's shape is(100L, 100L), while np.dot(enW, QTrans_esDiffTrans) is scalar
		# print "H: ", H
		H_list.append(H)
		sigmo_res = expit(H-1)

		# new_alpha = (1/iter_num) * alpha
		w_gradient = sigmo_res * en_vec_T.dot(esDiff_Qold)  # Ve x 1, 1 x Vs, Vs x k -> Ve x k
		Wnew = Wold - alpha * w_gradient

		q_gradient = sigmo_res * esDiff_T.dot(enW)   # Vs x k
		Qnew = Qold - alpha * q_gradient

		iter_num += 1
		iter_num2 += 1

		if is1stRun:
			is1stRun = False
			Wold = Wnew
			Qold = Qnew
			continue

		if iter_num2 % 500 == 0:
			L = sum(np.log(1 + np.exp(-H)) for H in H_list)
			print L
			L_list.append(L)
			iter_num2 = 0
		
		Wt1_Wt0 = LA.norm(np.subtract(Wnew, Wold))
		Wt0 = LA.norm(Wold)
		#print "||Wt+1 - Wt||: ", Wt1_Wt0
		#print "||Wt||: ", Wt0
		werror = Wt1_Wt0/Wt0 # Frobenius Norm: square root of the sum of the absolute squares of matrix elements

		Qt1_Qt0 = LA.norm(np.subtract(Qnew, Qold))
		Qt0 = LA.norm(Qold)
		#print "||Qt+1 - Qt||: ", Qt1_Qt0
		#print "||Qt||: ", Qt0
		qerror = LA.norm(np.subtract(Qnew, Qold))/LA.norm(Qold)
		# print iter_num, werror, qerror

		if werror < 0.000005 and qerror < 0.000005:
			isConverged = True
			break

		Wold = Wnew
		Qold = Qnew

	# Estimate the portion of train data used in the convergence
	used_portion = len(used_data) / train_size
	print "The number of used training data in SGD is " + str(len(used_data))
	print "%.2f" % used_portion + " of training data is used in SGD algorithm"

	plt.plot(L_list, [x*500 for x in range(0, int(len(L_list)/500))], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()