from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import datetime
import pickle

from quora_batcher import quora_data_batcher
from quora_model import quora_question_model

file_data = "./data/quora_duplicate_questions.tsv"
file_model = "./data/model.ckpt"
file_dic = "./data/dic.bin"
file_rdic = "./data/rdic.bin"
file_data_list = "./data/data_list.bin"
file_data_idx_list = "./data/data_idx_list.bin"
file_data_idx_list_test = "./data/data_idx_list_test.bin"
file_max_len = "./data/data_max_len.bin"
dir_summary = "./model/summary/"

np.random.seed(0)

print("-"*70)
print("QUORA ATTENTION TESTER..")
print("-"*70)
print()

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(now)
print("Load vocabulary from model file...")

with open(file_data_list, 'rb') as handle:
    data_list = pickle.load(handle)
with open(file_data_idx_list, 'rb') as handle:
    data_idx_list = pickle.load(handle)
with open(file_data_idx_list_test, 'rb') as handle:
    data_idx_list_test = pickle.load(handle)
with open(file_rdic, 'rb') as handle:
    rdic = pickle.load(handle)
with open(file_dic, 'rb') as handle:
    dic = pickle.load(handle)
with open(file_max_len, 'rb') as handle:
    max_len = pickle.load(handle)

print("data_list example")
print("question1 : ", end="")
print(data_list[int(len(data_list)//2)][0])
print("question2 : ", end="")
print(data_list[int(len(data_list)//2)][1])
print("target:%d" % data_list[int(len(data_list)//2)][2])
print()

print("data_list size = %d" % len(data_list))

SIZE_VOC = len(dic)
print("voc_size = %d" % SIZE_VOC)

SIZE_SENTENCE_MAX = max_len
print("max_sentence_len = %d" % SIZE_SENTENCE_MAX)
print()

print("dataset for train = %d" % len(data_idx_list))
print("dataset for test = %d" % len(data_idx_list_test))
SIZE_TRAIN_DATA = len(data_idx_list)
SIZE_TEST_DATA = len(data_idx_list_test)
print()

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(now)
print("Test start!!")
print()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:      
    batcher = quora_data_batcher(data_idx_list, data_idx_list_test, dic, SIZE_SENTENCE_MAX)
    
    model = quora_question_model(voc_size= SIZE_VOC, 
                                 target_size= 2,
                                 input_len_max= SIZE_SENTENCE_MAX, 
                                 lr= 0.0003,
                                 dev= "/cpu:0", 
                                 sess= sess,
                                 makedir= False)
    model.load_model(tf.train.latest_checkpoint("./model/2017-01-01 01:01/checkpoints/"))
    
    BATCHS_TEST = 500
    loop_cnt = SIZE_TEST_DATA // BATCHS_TEST
    last_cnt = BATCHS_TEST
    if SIZE_TEST_DATA % BATCHS_TEST > 0:
        loop_cnt += 1
        last_cnt = SIZE_TEST_DATA % BATCHS_TEST
    
    # remember all data result..
    # [all data(not equal), correct(not equal)]
    # [all data(equal), correct(equal)]
    table = np.zeros((2, 2), dtype=np.int)
    
    for loop in range(loop_cnt):
        if loop == loop_cnt-1:
            batchs = last_cnt
        else :
            batchs = BATCHS_TEST
        pos = loop * BATCHS_TEST
            
        data_x1, data_x2, data_y, len_x1, len_x2 =  batcher.get_test_batch_step(pos, batchs)

        results = model.batch_test(batchs, data_x1, data_x2, data_y, len_x1, len_x2, False)
        batch_pred = results[0]
        batch_loss = results[1]
        batch_acc = results[2]
        batch_att1 = results[3]
        batch_att2 = results[4]
        g_step = results[5]
        batch_lr = results[6]
        
        if loop % 100:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print("test_loss: %.4f, test_acc: %.4f (%s)" % (batch_loss, batch_acc, now))
            print(table)
            print()
        
        for b in range(batchs):
            target = data_y[b]
            predic = batch_pred[b]

            current_cnt = table[target][0]
            current_ok = table[target][1]

            table[target][0] = current_cnt+1
            if target==predic:
                table[target][1] = current_ok+1
        
    print(table)
    
    """"""
    # visualize..
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import rcParams, rc
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    path = '/home/jupyter/cuteboydot/fonts/AppleGothic.ttf'
    prop =  fm.FontProperties(fname=path)
    path = fm.findfont(prop, directory=path)
    print(prop.get_name())
    rc('font', family=prop.get_name())
    rc('text', usetex='false')
    rcParams['font.family'] = prop.get_name()
    rcParams.update({'font.size': 14})
    print()

    for aa in range(min(100, batchs)):
        attend1 = batch_att1[aa]
        attend1 = np.reshape(attend1, (1, -1))
        sentence1 = [rdic[w] for w in data_x1[aa] if w != 0]
        
        attend2 = batch_att2[aa]
        attend2 = np.reshape(attend2, (1, -1))
        sentence2 = [rdic[w] for w in data_x2[aa] if w != 0]

        print(sentence1)
        print(sentence2)
        print("target:%d, Predict:%d" % (data_y[aa], batch_pred[aa]))

        print("attend1")
        plt.clf()
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(attend1[:, :len(sentence1)], cmap="YlOrBr")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xticks(range(len(sentence1)))
        ax.set_xticklabels(sentence1, fontsize=14, rotation=90, fontproperties=prop)
        ax.set_yticks(range(1))
        ax.set_yticklabels(["prob "], fontsize=14, rotation=0, fontproperties=prop)

        ax.grid()
        plt.show()
           
        print("attend2")
        plt.clf()
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(attend2[:, :len(sentence2)], cmap="YlOrBr")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xticks(range(len(sentence2)))
        ax.set_xticklabels(sentence2, fontsize=14, rotation=90, fontproperties=prop)

        ax.set_yticks(range(1))
        ax.set_yticklabels(["prob "], fontsize=14, rotation=0, fontproperties=prop)

        ax.grid()
        plt.show()
        
        print("~" * 70)

print()
print("Test finished!!")
print()