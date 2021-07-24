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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

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
    output_file_dir = os.path.join(output_dir, args.save_results_path)
    model_dir = os.path.join(output_dir, args.model_dir)
    print(output_file_dir)
    for di in [model_dir, output_file_dir]:
        if not os.path.exists(di):
            os.makedirs(di)
            
    return model_dir, output_file_dir

def check_inputs(args):
    check_list_labeled_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    check_list_known_cls_ratios = [0.25, 0.5, 0.75] 
    if args.labeled_ratio not in check_list_labeled_ratios:
        print('The assigned labeled ratio is unavailable!')
        return False
    if args.known_cls_ratio not in check_list_known_cls_ratios:
        print('The assigned known class ratio is unavailable!')
        return False

    return True

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
    if args.train_detect:
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
                        values = _list[ (_list[sort_type] == item) ][metric]
                        val[pos_map[str(item)]] =  '%.2f' %  ( values.mean() )
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

def restore_loss(loss, model_dir):
    output_loss_file = os.path.join(self.model_dir, 'loss', WEIGHTS_NAME)
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
    results_path = os.path.join(args.train_data_dir, args.type, result_file)
    
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

def PCA_reduce_feats(feats, dim):
    estimator = PCA(n_components=dim)
    print("utils.py: 402: ", feats.shape)
    reduce_feats = estimator.fit_transform(feats)
    
    return reduce_feats
def TSNE_reduce_feats(feats, dim):
    estimator = TSNE(n_components=dim)
    # estimator = TSNE(n_components=2, n_iter=8000, learning_rate=10, n_iter_without_progress=1200)
    print("utils.py: 409: ", feats.shape)
    print("utils.py: 410: ", feats.shape)
    reduce_feats = estimator.fit_transform(feats)
    
    return reduce_feats

def UMAP_reduce_feats(feats):
    estimator = umap.UMAP(random_state=42)
    reduce_feats = estimator.fit_transform(feats)
    return reduce_feats

def twoD_draw_centers_radius(args, data, outputs):
    
    model_dir, output_file_dir = set_path(args)
    centroids = np.load(os.path.join(output_file_dir, 'centroids.npy'))
    deltas = np.load(os.path.join(output_file_dir, 'deltas.npy'))
    # deltas = [round(x, 2) for x in delta_tmp]

    predictions = list([data.label_list[idx] for idx in outputs[0]]) 
    true_labels = list([data.label_list[idx] for idx in outputs[1]]) 
    feats = outputs[2]

    # reduce_centers = TSNE_reduce_feats(centroids, 2)
    # reduce_centers = [round(x, 2) for x in reduce_centers]
    # reduce_feats = TSNE_reduce_feats(feats, 2)
    # reduce_centers = []
    # labels = np.unique(outputs[0])
    # for label in labels:
    #     print(data.label_list[label])
    #     pos = list(np.where(outputs[0] == label)[0])
    #     print('pos', pos)
    #     center = np.mean(reduce_feats[pos], axis = 1)
    #     print('center', center)
    #     reduce_centers.append(center)
    # reduce_centers = PCA_reduce_feats(centroids, 2)
    # reduce_centers = [np.round(x, 2) for x in reduce_centers]
    # reduce_feats = PCA_reduce_feats(feats, 2)
    reduce_centers = TSNE_reduce_feats(centroids, 2)
    reduce_centers = [np.round(x, 2) for x in reduce_centers]
    reduce_feats = TSNE_reduce_feats(feats, 2)
    


    # print('reduce_feats', reduce_feats.shape)
    # print('reduce_center', reduce_centers.shape)
    # print('centroids', centroids.shape)

    static_dir = os.path.join(args.frontend_dir, args.type)
    draw_center_r_path = os.path.join(static_dir, args.method + '_analysis.json')
    
    all_dict = {}

    boundaries = {}

    for idx in range(len(reduce_centers)):
        center = reduce_centers[idx]
        boundary = deltas[idx]
        # print('center', center)
        # print('boundary', boundary)
        elem = list(center)
        elem.append(boundary)
        elem = [ round(float(x), 2) for x in elem]
        
        label = data.label_list[idx] + '_boundary'
        boundaries[label] = elem
    
    # print('boundaries', boundaries)
    name = 'boundary_' + args.dataset
    all_dict[name] = boundaries
    # print('all_dict', all_dict)

    points = {}
    # reduce_feats = PCA_reduce_feats(feats)
    reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2) ] for item in reduce_feats ]

    # point_list = []
    for idx in range(len(reduce_centers)):
        pos = list(np.where(outputs[0] == idx)[0])

        label_item = data.label_list[idx]
        center_msg_item = boundaries[label_item + "_boundary"]
        samples = []
        for i, feat in enumerate(reduce_feats):
            if i in pos:
                # feat.extend(center_msg_item)
                #lee-mdoify
                # feat = [ feat + center_msg_item ]
                #lee-mdoify

                feat = feat +center_msg_item
                
                samples.append(feat)
        # samples = [feat.extend(center_msg_item) for idx, feat in enumerate(reduce_feats) if idx in pos]
        print("utils.py: 505: ", samples)
        points[label_item] = samples
        # points[label] = [samples] + [label] + boundaries[label+"_boundary"][0]
        # point_list.append([samples + [label]])
    
    # for key in points.keys():
    #     print('key', key)
    #     print('point', points[key])

    # print(points)
    name = 'points_' + args.dataset
    all_dict[name] = points
    # all_dict["point_list_"+args.dataset] = point_list
    print("500:\t", points)
    json_add(all_dict, draw_center_r_path)

# def threeD_draw_centers_radius(args, data, outputs):
    
#     model_dir, output_file_dir = set_path(args)
#     centroids = np.load(os.path.join(output_file_dir, 'centroids.npy'))
#     deltas = np.load(os.path.join(output_file_dir, 'deltas.npy'))
#     # deltas = [round(x, 2) for x in delta_tmp]

#     predictions = list([data.label_list[idx] for idx in outputs[0]]) 
#     true_labels = list([data.label_list[idx] for idx in outputs[1]]) 
#     feats = outputs[2]

#     reduce_centers = PCA_reduce_feats(centroids, 3)
#     # reduce_centers = [round(x, 2) for x in reduce_centers]
#     reduce_feats = PCA_reduce_feats(feats, 3)
    
#     # reduce_centers = TSNE_reduce_feats(centroids)
#     # # reduce_centers = [round(x, 2) for x in reduce_centers]
#     # reduce_feats = TSNE_reduce_feats(feats)
    


#     print('reduce_feats', reduce_feats.shape)
#     print('reduce_center', reduce_centers.shape)
#     # print('centroids', centroids.shape)

#     static_dir = os.path.join(args.frontend_dir, args.type)
#     draw_center_r_path = os.path.join(static_dir, args.method + '_analysis_3D.json')
    
#     all_dict = {}

#     boundaries = {}

#     for idx in range(len(reduce_centers)):
#         center = reduce_centers[idx]
#         boundary = deltas[idx]
#         # print('center', center)
#         # print('boundary', boundary)
#         elem = list(center)
#         elem.append(boundary)
#         elem =  [round(float(x), 2) for x in elem] 
        
#         label = data.label_list[idx] + '_boundary'
#         boundaries[label] = elem
    
#     # print('boundaries', boundaries)
#     name = 'boundary_' + args.dataset
#     all_dict[name] = boundaries
#     # print('all_dict', all_dict)

#     points = {}
#     reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2), round(float(item[1]), 2)] for item in reduce_feats]


#     for idx in range(len(reduce_centers)):
#         pos = list(np.where(outputs[0] == idx)[0])
#         samples = [feat for idx, feat in enumerate(reduce_feats) if idx in pos]

#         label = data.label_list[idx]
#         points[label] = samples
    
#     # for key in points.keys():
#     #     print('key', key)
#     #     print('point', points[key])

#     # print(points)
#     name = 'points_' + args.dataset
#     all_dict[name] = points
#     json_add(all_dict, draw_center_r_path)

def get_probs(args, data, outputs):
    
    interval = 0.1
    preds, trues, probs = outputs[0], outputs[1], outputs[2]
    confidences = []
    score = 0
    while True:
        score += interval

        if score >= 1:
            break
        confidences.append(round(score, 2))

    print('confs', confidences)
    print(probs.shape)

    all_dicts = {}

    known_intents = []
    open_intents = []

    for conf in confidences:
        up_score = conf
        low_score = conf - interval
        pos = [idx for idx, prob in enumerate(probs) if (prob >= low_score and prob <= up_score)]
        num_knowns = len([p for p in pos if outputs[1][p] != data.unseen_token_id])
        num_opens = -len([p for p in pos if outputs[1][p] == data.unseen_token_id])

        known_intents.append(num_knowns)
        open_intents.append(num_opens)
    
    static_dir = os.path.join(args.frontend_dir, args.type)
    path = os.path.join(static_dir, args.method + '_analysis.json')

    name = args.method + '_' + args.dataset
    
    num_dict = {}
    num_dict['Known_Intent'] = known_intents
    num_dict['Open_Intent'] = open_intents

    xais_name = name + '_x'
    xais_dict = {}
    xais_dict['xais'] = confidences

    all_dicts[name] = num_dict
    all_dicts[xais_name] = xais_dict

    json_add(all_dicts, path)


# def save_detect_frontend_results(manager, args, data, outputs):

#     results_path = os.path.join(args.train_data_dir, args.type, 'results.csv')
    
#     static_dir = os.path.join(args.frontend_dir, args.type)
#     if not os.path.exists(static_dir):
#         os.makedirs(static_dir)

#     #save true_false predictions
#     predictions = list([data.label_list[idx] for idx in outputs[0]]) 
#     true_labels = list([data.label_list[idx] for idx in outputs[1]]) 
#     predict_t_f, predict_t_f_fine = cal_true_false(true_labels, predictions)
#     csv_to_json(results_path, static_dir)

#     tf_overall_path = os.path.join(static_dir, 'true_false_overall.json')
#     tf_fine_path = os.path.join(static_dir, 'true_false_fine.json')

#     results = {}
#     results_fine = {}
#     key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.labeled_ratio) + '_' + str(args.method)
#     if os.path.exists(tf_overall_path):
#         results = json_read(tf_overall_path)

#     results[key] = predict_t_f

#     if os.path.exists(tf_fine_path):
#         results_fine = json_read(tf_fine_path)
#     results_fine[key] = predict_t_f_fine

#     json_add(results, tf_overall_path)
#     json_add(results_fine, tf_fine_path)

#     # print('test_results', data_diagram)

# def save_detect_table_results_to_frontend(args, data, outputs):
#     test_trues = list([data.label_list[idx] for idx in outputs[1]]) 
#     test_preds = list([data.label_list[idx] for idx in outputs[0]]) 
#     # test_trues = np.array([data.label_list[idx] for idx in outputs[1]]) 
#     # test_preds = np.array([data.label_list[idx] for idx in outputs[0]]) 
#     test_texts = [example.text_a for example in data.test_examples]
#     # print('\n\n\ntrue:',len(test_trues),'\npred:',len(test_preds),'\n\ntrue:',outputs[1],'\n\npred:',outputs[0])
#     # # print('\n\n\ntrue:',len(test_true_label_list),'\npred:',len(test_preds),'\n\ntrue:',test_true_label_list,'\n\npred:',test_preds)
#     # print('\n\n\ntrue:',len(test_trues),'\npred:',len(test_preds),'\n\ntrue:',test_trues,'\n\npred:',test_preds)

#     # ids_known = [idx for idx, label in enumerate(outputs[0]) if label != data.unseen_token_id]
#     # known_label_list = list([ example.label for i, example in enumerate(data.train_examples) if i in ids_known ])
#     known_label_list = test_preds.copy()
#     known_label_list_unique = list(dict.fromkeys(known_label_list))
#     dataset_info = {}
#     dataset_info_json_path = os.path.join(sys.path[0], '../frontend/static/jsons/open_intent_detection', 'analysis_table_info.json')
#     if os.path.exists(dataset_info_json_path):
#         dataset_info = json_read(dataset_info_json_path)
#     class_list = []
#     known_num = 0
#     for known_label_item in known_label_list_unique:
#         text_list = []
#         text_true_list_tmp = []
#         # 获取等于当前label的
#         # known_label_item_ids = list([ np.where( known_label_item ==  test_preds)[0] ])
#         known_label_item_ids = list([ i for i,x in enumerate(known_label_list) if x == known_label_item ])
#         known_num += len(known_label_item_ids)
#         # print('=='*20,'\n\n',known_label_item, '\t', len(known_label_item_ids))
#         for i in known_label_item_ids :
#             text_true_list_tmp.append(test_trues[i])
#             text_list.append({
#                 "dataset_name":args.dataset, "class_type":'known',
#                 "label_name": test_trues[i],
#                 "method": args.method,
#                 "text": test_texts[i]
#             })
#         class_list.append({"label_name": known_label_item, "label_text_num":len(known_label_item_ids),
#             "dataset_name":args.dataset, "method": args.method, "class_type":'known'})
#         # print('=='*20,'\n\n',known_label_item, '\t\n', text_true_list_tmp,'\n\n','=='*20)
#         # add text_list to dataset_info
#         dataset_info['text_list_'+args.dataset+"_"+args.method+"_known_"+known_label_item] = text_list
#     # add class_list to dataset_info
#     dataset_info["class_list_"+args.dataset+"_"+args.method+"_known"] = class_list
#     # save dataset_info to frontend_file
#     json_add(dataset_info, dataset_info_json_path)
#     # with open(dataset_info_json_path, 'w') as write_f:
#     #     json.dump(dataset_info, write_f, indent=4)