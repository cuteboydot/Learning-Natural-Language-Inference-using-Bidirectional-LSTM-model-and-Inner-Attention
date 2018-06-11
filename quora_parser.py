from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import random
import re

import numpy as np
import pandas as pd
from contractions import contraction
from word2vec import Word2Vec

file_data = "./data/quora_duplicate_questions.tsv"
file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_list = "./data/data_list.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


df = pd.read_csv(file_data, sep="\t")
df = df.dropna()
print("data frame set shape : %s" % (str(df.shape)))
print()

q_map = np.zeros((300), np.int32)
data_list0 = []
data_list1 = []
word_list = []
q1_max = 0
q2_max = 0
for index, row in df.iterrows():
    q1 = row[3]
    q2 = row[4]
    ans = int(row[5])

    q1 = clean_str(q1)
    q2 = clean_str(q2)

    word_list.append(q1)
    word_list.append(q2)

    q1 = q1.split()
    q2 = q2.split()

    q_map[len(q1)] += 1
    q_map[len(q2)] += 1

    if len(q1) < 2:
        continue
    if len(q2) < 2:
        continue

    if len(q1) > 100:
        continue
    if len(q2) > 100:
        continue

    if q1_max < len(q1):
        q1_max = len(q1)
    if q2_max < len(q2):
        q2_max = len(q2)

    if ans == 0:
        data_list0.append([q1, q2, ans])
    if ans == 1:
        data_list1.append([q1, q2, ans])

for zzz in range(300):
    if q_map[zzz] > 0:
        print("[%03d] count : %d" % (zzz, q_map[zzz]))

print("q1_max : %d, q2_max : %d" % (q1_max, q2_max))
print("data_list0 size : %d" % (len(data_list0)))
print("data_list0 example")
for a in range(2):
    print(data_list0[a])
print()
print("data_list1 size : %d" % (len(data_list1)))
print("data_list1 example")
for a in range(2):
    print(data_list1[a])
print()

total_words = " ".join(word_list).split()
count = collections.Counter(total_words).most_common()

symbols = ["<PAD>", "<UNK>"]
rdic = symbols + [i[0] for i in count if i[1] > 0]     # word list order by count desc
dic = {w: i for i, w in enumerate(rdic)}                      # dic {word:count} order by count desc
voc_size = len(dic)
print("voc_size size = %d" % voc_size)
print("data_list example")
print(rdic[:20])
SIZE_VOC = len(dic)

data_idx_list0 = []
data_idx_list1 = []
for q1, q2, ans in data_list0:
    q1_idx = []
    q2_idx = []

    for word in q1:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        q1_idx.append(idx)

    for word in q2:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        q2_idx.append(idx)

    data_idx_list0.append([q1_idx, q2_idx, ans])

for q1, q2, ans in data_list1:
    q1_idx = []
    q2_idx = []

    for word in q1:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        q1_idx.append(idx)

    for word in q2:
        idx = -1
        if word in dic:
            idx = dic[word]
        else:
            idx = dic["<UNK>"]
        assert 0 <= idx < SIZE_VOC
        q2_idx.append(idx)

    data_idx_list1.append([q1_idx, q2_idx, ans])

print("data_idx_list0 size : %d" % (len(data_idx_list0)))
print("data_idx_list0 example")
for a in range(2):
    print(data_idx_list0[a])
print()
print("data_idx_list1 size : %d" % (len(data_idx_list1)))
print("data_idx_list1 example")
for a in range(2):
    print(data_idx_list1[a])
print()

#max_len = max(q1_max, q2_max)
max_len = 100
print("sentence max len = %d" % max_len)
print()

random.shuffle(data_idx_list0)
random.shuffle(data_idx_list1)

test0 = int(len(data_idx_list0) * 0.1)
test1 = int(len(data_idx_list1) * 0.1)

data_idx_list_test = data_idx_list0[:test0] + data_idx_list1[:test1]
data_idx_list = data_idx_list0[test0:] + data_idx_list1[test1:]
random.shuffle(data_idx_list_test)
random.shuffle(data_idx_list)
print("dataset for train = %d" % len(data_idx_list))
print("dataset for test = %d" % len(data_idx_list_test))
print()


# save dictionary
with open(file_data_idx_list, 'wb') as handle:
    pickle.dump(data_idx_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_data_idx_list_test, 'wb') as handle:
    pickle.dump(data_idx_list_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_dic, 'wb') as handle:
    pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_rdic, 'wb') as handle:
    pickle.dump(rdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_max_len, 'wb') as handle:
    pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("dictionary files saved..")
print()



