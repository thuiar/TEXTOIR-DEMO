import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import copy
import random
import csv
import sys
import math
import json
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.nn.utils import weight_norm

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
    if args.train:
        manager_attrs = ["device","best_eval_score","test_results"]
    else:
        manager_attrs = ["device", "test_results"]

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


def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['F1-known'] = f_seen
    result['F1-open'] = f_unseen
    result['F1'] = f
        
    return result

def plot_confusion_matrix(cm, classes, save_name, normalize=False, title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues, save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.switch_backend('agg')
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
    if save:
        plt.savefig(save_name)
    
def draw(x, y):
    from matplotlib.colors import ListedColormap
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    print("TSNE: fitting start...")
    tsne = TSNE(2, n_jobs=4, perplexity=30)
    Y = tsne.fit_transform(x)

    # matplotlib_axes_logger.setLevel('ERROR')
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
#     labels = ['wordpress','oracle','svn','apache','excel','matlab','visual-studio','cocoa','osx','bash','unknown']
    id_to_label = {i: label for i, label in enumerate(labels) }
    y_true = pd.Series(y)
    plt.style.use('ggplot')
    n_class = y_true.unique().shape[0]
    colors = ( 'gray','lightgreen', 'plum','DarkMagenta','SkyBlue','PaleTurquoise','DeepPink','Gold','Orange','Brown','DarkKhaki')

    #cmap = plt.cm.get_cmap("tab20", n_class)

    fig, ax = plt.subplots(figsize=(9, 6), )
    la = [i for i in range(n_class)]
    la = sorted(la,reverse=True)
    cmap = ListedColormap(colors)
    for idx, label in enumerate(la):
        ix = y_true[y_true==label].index
        x = Y[:, 0][ix]
        y = Y[:, 1][ix]
        ax.scatter(x, y, c=cmap(idx), label=id_to_label[label], alpha=0.5)
    #     ax.scatter(x, y, c=np.random.rand(3,), label=label, s=100)

    # Shrink current axis by 20%
    ax.set_title('proto_loss')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('output/tsne-CDAC+2.pdf', bbox_inches='tight')
#     plt.show()

def plot_curve(points):
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', '|', '_', '+', 'h', 'H', '.', ',', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = [i for i in range(len(centers[0]))]
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
        
    plt.xlabel('Epoch')
    plt.ylabel('Decision Boundary $\Delta$')
    plt.legend()
    plt.title('50% Known Classes on StackOverflow')
    plt.show()
    plt.savefig('curve.pdf')

def produce_json(df, method, dataset, select_type, sort_type, select_terms, metricList):
    import csv
    if sort_type == 'known_cls_ratio':
        axis_len = 3
        pos_map = {'0.25':0, '0.5':1, '0.75':2}
    elif sort_type == 'labeled_ratio':
        axis_len = 5
        pos_map = {'0.2':0, '0.4':1, '0.6':2, '0.8':3, '1.0':4}
        
    dic = {}
    for i, dataset in dataset.items():
        for metric in metricList:
            for j, select_term in select_terms.items():
                dic_tmp = {}
                for k, method_val in method.items():
                    _list = df[ (df["dataset"].str[:]==(dataset)) & (df[select_type] == select_term) & (df["method"].str[:]==(method_val))  ].sort_values(sort_type)
                    select_tmp = _list.drop_duplicates(subset=[sort_type],keep='first')[sort_type]
                    val=[0] * axis_len
                    
                    for l,item in select_tmp.items():
                        val[pos_map[str(item)]] = _list[ (_list[sort_type] == item) ][metric].mean()
                    dic_tmp[method_val]=val
                # print(dic_1)
                dic['detection_'+str(dataset)+'_'+str(select_term)+'_'+str(metric)] = dic_tmp

    return dic

def csv_to_json(csv_file, frontend_dir):
    df = pd.read_csv(csv_file)

    dataset = df.drop_duplicates(subset=['dataset'],keep='first')['dataset']
    known_cls_ratio = df.drop_duplicates(subset=['known_cls_ratio'],keep='first')['known_cls_ratio']
    labeled_ratio = df.drop_duplicates(subset=['labeled_ratio'],keep='first')['labeled_ratio']
    method = df.drop_duplicates(subset=['method'],keep='first')['method']   

    metricList=['F1','F1-known','F1-open', 'Acc']

    select_types = ['known_cls_ratio', 'labeled_ratio']
    select_terms = [known_cls_ratio, labeled_ratio]
    select_files = ['json_detection_IOKIR.json','json_detection_IOLR.json' ]

    for i in range(len(select_types)):
        select_type = select_types[i]
        sort_type = select_types[(i + 1) % 2]
        select_term = select_terms[i]
        select_file = select_files[i] 
        
        dic = produce_json(df, method, dataset, select_type,  sort_type, select_term, metricList)
        select_path = os.path.join(frontend_dir, select_files[(i + 1) % 2] )
        with open(select_path,'w+') as f:
            json.dump(dic,f,indent=4)
        
def json_read(path):
    
    with open(path, 'r')  as f:
        json_r = json.load(f)

    return json_r

def json_add(predict_t_f, path):
    
    with open(path, 'w') as f:
        json.dump(predict_t_f, f, indent=4)

def cal_true_false(true_labels, predictions):
            
    results = {"intent_class":[], "left":[], "right":[]}
    trues = np.array(true_labels)
    preds = np.array(predictions)

    labels = np.unique(trues)

    results_fine = {}
    label2id = {x:i for i,x in enumerate(labels)}

    for label in labels:
        pos = np.array(np.where(trues == label)[0])
        num_pos = int(np.sum(preds[pos] == trues[pos]))
        num_neg = int(np.sum(preds[pos] != trues[pos]))

        results["intent_class"].append(label)
        results["left"].append(-num_neg)
        results["right"].append(num_pos)

        tmp_list = [0] * len(labels)
        
        for fine_label in labels:
            if fine_label != label:
                
                num = int(np.sum(preds[pos] == fine_label))
                tmp_list[label2id[fine_label]] = num
                
        results_fine[label] = tmp_list

    return results, results_fine

