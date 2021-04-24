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
import importlib
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.utils import weight_norm

def get_manager(args, data):

    if not os.path.exists(os.path.join(args.train_data_dir, args.type)):
        os.makedirs(os.path.join(args.train_data_dir, args.type))

    module_names = [args.type, 'methods', args.method, 'manager']
    import_name = ".".join(module_names)
    method = importlib.import_module(import_name) 
    manager = method.ModelManager(args, data)   

    return manager

def set_path(args):
    
    concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
    output_file_name = "_".join([str(x) for x in concat_names])
    output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
    output_file_dir = os.path.join(output_dir, args.output_path)
    model_dir = os.path.join(output_dir, args.model_dir)
    results_dir = os.path.join(args.results_path, args.type)
    
    for di in [model_dir, output_file_dir, results_dir]:
        if not os.path.exists(di):
            os.makedirs(di)
            
    return model_dir, output_file_dir

def debug(outputs, data, manager, args):

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
    if args.train_detect:
        manager_attrs = ["device","best_eval_score","test_results"]
    else:
        manager_attrs = ["device", "test_results"]

    for attr in manager_attrs:
        attr_name = attr
        attr_value = manager.__getattribute__(attr)
        print(attr_name,':',attr_value)
    
    y_true, y_pred = outputs[0], outputs[1]
    predictions = list([data.label_list[idx] for idx in y_pred]) 
    true_labels = list([data.label_list[idx] for idx in y_true]) 
    print('-----------------Prediction Example--------------------')
    show_num = 10
    for i,example in enumerate(data.test_examples):
        if i >= show_num:
            break
        sentence = example.text_a
        true_label = true_labels[i]
        predict_label = predictions[i]
        print(i,':',sentence)
        print('Pred: {}; True: {}'.format(predict_label, true_label))


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

def cal_detection_results(test_results, show=False):
    """
    test_results[0]: true labels
    test_results[1]: pred labels
    test_results[2]: extracted features
    test_results[3]: predicted probabilities
    """
    y_true, y_pred = test_results[0], test_results[1]
    cm = confusion_matrix(y_true, y_pred)
    results = F_measure(cm)
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    results['Acc'] = acc
    
    if show:
        print('cm',cm)
        print('results', results)
    
    return results

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def save_loss(loss_fct, model_dir):
    loss_dir = os.path.join(model_dir, 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    save_loss = loss_fct.module if hasattr(loss_fct, 'module') else loss_fct
    loss_file = os.path.join(loss_dir, WEIGHTS_NAME)
    torch.save(save_loss.state_dict(), loss_file)

def restore_loss(loss_fct, model_dir):
    output_loss_file = os.path.join(model_dir, 'loss', WEIGHTS_NAME)
    loss_fct.load_state_dict(torch.load(output_loss_file))
    return loss_fct

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def freeze_bert_parameters(model):
    for name, param in model.bert.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model

def save_detect_backend_results(manager, args, data):

    np.save(os.path.join(manager.output_file_dir, 'labels.npy'), data.label_list)

    var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
    names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(manager.test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    result_file = 'results.csv'
    results_path = os.path.join(args.results_path, args.type, result_file)
    
    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

###################################Draw############################
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
    tsne = TSNE(2, n_jobs=4, perplexity=100)
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
