# -*- coding:utf-8 -*-
import itertools
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import torch.nn.functional as F
import random
import csv
import sys
import logging
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def debug(data, manager, args):

    # args_attrs = ["max_seq_length","feat_dim","warmup_proportion","freeze_bert_parameters","task_name",
    #               "known_cls_ratio","labeled_ratio","method","seed","gpu_id","num_train_epochs","lr",
    #               "train_batch_size","eval_batch_size","wait_patient","threshold"]

    print('-----------------Data--------------------')
    data_attrs = ["data_dir","n_known_cls","num_labels","all_label_list","known_label_list"]

    for attr in data_attrs:
        attr_name = attr
        attr_value = data.__getattribute__(attr)
        print(attr_name,':',attr_value)

    print('-----------------Args--------------------')
    for k in list(vars(args).keys()):
        print(k,':',vars(args)[k])

    print('-----------------Manager--------------------')
    manager_attrs = ["device","best_eval_score","test_results"]

    for attr in manager_attrs:
        attr_name = attr
        attr_value = manager.__getattribute__(attr)
        print(attr_name,':',attr_value)
    
    if manager.predictions is not None:
        print('-----------------Prediction Example--------------------')
        show_num = 10
        for i,example in enumerate(data.test_examples):
            if i >= show_num:
                break
            sentence = example.text_a
            true_label = manager.true_labels[i]
            predict_label = manager.predictions[i]
            print(i,':',sentence)
            print('Pred: {}; True: {}'.format(predict_label,true_label))

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}


####################################Unsupervised Utils#######################################################
import re
from collections import defaultdict
import logging
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import seaborn as sns
import tensorflow as tf
import random as rn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Modeling
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.optimizers import SGD
from keras.engine.topology import Layer, InputSpec
# Evaluation
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder

# Visualization
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def set_allow_growth(device):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list = device
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess) # set this TensorFlow session as the default session for Keras


def create_logger(app_name="root", level=logging.DEBUG):
    # 基礎設定
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler('logs/' + app_name + '.log', 'w', 'utf-8'), ])

    # 定義 handler 輸出 sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)

    # handler 設定輸出格式
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(app_name)
    return logger


def plot_cluster_pie(results, k, n_cols, df_plot, y_pred):
    NMI = results['NMI']
    ACC = results['ACC']
    n_rows = int(np.ceil(k/n_cols))

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/k) for i in range(k)]

    d = {val:key for key, val in enumerate(df_plot.label.unique())}
    df_plot['y_pred'] = y_pred
    fig, axes = plt.subplots(nrows=n_rows,  ncols=n_cols, figsize=(round(3.5*n_cols), 4*n_rows))
    for i in tqdm(range(k)):
        x = i//n_cols
        y = i%n_cols

        df_ = df_plot[df_plot.y_pred==i]
        y_true_vc = df_.label.value_counts()
        if y_true_vc.size==0:
            print("Encounter empty cluster, ignore")
            continue
            
        explode = [0] * len(y_true_vc.values)
        explode[0] = 0.1
        labels = y_true_vc.index
        cs = [colors[d[label]] for label in labels]
        axes[x, y].pie(y_true_vc.values, explode=explode, labels=labels, colors=cs)

        label_majority = y_true_vc.index[0]
        percentage = y_true_vc.values[0]/sum(y_true_vc) * 100
        n_samples = df_.shape[0]
        title = "%d: #=%d, %s(%1.1f%%)" % (i, n_samples, label_majority,  percentage)
        axes[x, y].set_title(title)

        centre_circle = plt.Circle((0,0), 0.50, fc='white')
        axes[x, y].add_artist(centre_circle)
        fig.tight_layout()
    fig.suptitle("K=%d (ACC=%1.2f, NMI=%1.2f)" % (k, ACC, NMI), fontsize=24)
    fig.subplots_adjust(top=0.9)

def plot_cluster_wordcloud(results, k, n_cols, df_plot, y_pred):
    NMI = results['NMI']
    ACC = results['ACC']
    n_rows = int(np.ceil(k/n_cols))
    stop_words = set(stopwords.words('english') + ['.', '\'', ','])
    fig, axes = plt.subplots(nrows=n_rows,  ncols=n_cols, figsize=(round(3.5*n_cols), 4*n_rows))
    for i in tqdm(range(k)):
        x = i//n_cols
        y = i%n_cols
        cnt = Counter()
        df_ = df_plot[df_plot.y_pred==i]
        y_true_vc = df_.label.value_counts()
        if y_true_vc.size==0: # 
            print("Encounter empty cluster, ignore")
            continue
            
        label_majority = y_true_vc.index[0]
        percentage = y_true_vc.values[0]/sum(y_true_vc) * 100

        for sentence in df_['words'].tolist():
            for word in sentence:
                if word not in stop_words:
                    cnt[word] += 1
        wordcloud = WordCloud(width=400, height=400, relative_scaling=0.5, normalize_plurals=False, 
                              background_color='white'
                             ).generate_from_frequencies(cnt)
        n_samples = df_.shape[0]
        title = "%d: #=%d, %s(%1.1f%%)" % (i, n_samples, label_majority,  percentage)
        axes[x, y].imshow(wordcloud, interpolation='bilinear')
        axes[x, y].set_title(title)

        centre_circle = plt.Circle((0,0),0.50, fc='white')
        axes[x, y].add_artist(centre_circle)
        axes[x, y].axis("off")
        fig.tight_layout()
    fig.suptitle("K=%d (ACC=%1.2f, NMI=%1.2f)" % (k, ACC, NMI), fontsize=24)
    fig.subplots_adjust(top=0.9)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12,10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('result.png')
    


