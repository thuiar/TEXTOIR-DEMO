import pandas as pd 
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap                                                                                                                           
import numpy as np
import csv
import os
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
    print("final!!!!!!")

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

# def PCA_reduce_feats(feats, dim):
#     estimator = PCA(n_components=dim)
#     print("utils.py: 402: ", feats.shape)
#     reduce_feats = estimator.fit_transform(feats)
    
#     return reduce_feats

def produce_json(df, method, dataset, select_type, sort_type, select_terms, metricList):
    
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

def twoD_draw_centers_radius(args, manager, data, outputs):
    
    model_dir, output_file_dir = manager.model_dir, manager.output_file_dir
    centroids = np.load(os.path.join(output_file_dir, 'centroids.npy'))
    deltas = np.load(os.path.join(output_file_dir, 'deltas.npy'))
    # deltas = [round(x, 2) for x in delta_tmp]

    predictions = list([data.label_list[idx] for idx in outputs[0]]) 
    true_labels = list([data.label_list[idx] for idx in outputs[1]]) 
    feats = outputs[2]
    reduce_centers = TSNE_reduce_feats(centroids, 2)
    reduce_centers = [np.round(x, 2) for x in reduce_centers]
    reduce_feats = TSNE_reduce_feats(feats, 2)

    static_dir = os.path.join(args.frontend_dir, args.type)
    draw_center_r_path = os.path.join(static_dir, args.method + '_analysis.json')
    
    all_dict = {}

    boundaries = {}

    for idx in range(len(reduce_centers)):
        center = reduce_centers[idx]
        boundary = deltas[idx]
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
    reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2) ] for item in reduce_feats]

    # point_list = []
    for idx in range(len(reduce_centers)):
        pos = list(np.where(y_pred == idx)[0])

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

def threeD_draw_centers_radius(args, data, outputs):
    
    y_true, y_pred, feats = outputs[0], outputs[1], outputs[2]

    model_dir, output_file_dir = set_path(args)
    centroids = np.load(os.path.join(output_file_dir, 'centroids.npy'))
    deltas = np.load(os.path.join(output_file_dir, 'deltas.npy'))
    # deltas = [round(x, 2) for x in delta_tmp]

    predictions = list([data.label_list[idx] for idx in y_pred]) 
    true_labels = list([data.label_list[idx] for idx in y_true]) 

    reduce_centers = PCA_reduce_feats(centroids, 3)
    # reduce_centers = [round(x, 2) for x in reduce_centers]
    reduce_feats = PCA_reduce_feats(feats, 3)
    
    # reduce_centers = TSNE_reduce_feats(centroids)
    # # reduce_centers = [round(x, 2) for x in reduce_centers]
    # reduce_feats = TSNE_reduce_feats(feats)
    


    print('reduce_feats', reduce_feats.shape)
    print('reduce_center', reduce_centers.shape)
    # print('centroids', centroids.shape)

    static_dir = os.path.join(args.frontend_dir, args.type)
    draw_center_r_path = os.path.join(static_dir, args.method + '_analysis_3D.json')
    
    all_dict = {}

    boundaries = {}

    for idx in range(len(reduce_centers)):
        center = reduce_centers[idx]
        boundary = deltas[idx]
        # print('center', center)
        # print('boundary', boundary)
        elem = list(center)
        elem.append(boundary)
        elem =  [round(float(x), 2) for x in elem] 
        
        label = data.label_list[idx] + '_boundary'
        boundaries[label] = elem
    
    # print('boundaries', boundaries)
    name = 'boundary_' + args.dataset
    all_dict[name] = boundaries
    # print('all_dict', all_dict)

    points = {}
    reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2), round(float(item[1]), 2)] for item in reduce_feats]


    for idx in range(len(reduce_centers)):
        pos = list(np.where(y_pred == idx)[0])
        samples = [feat for idx, feat in enumerate(reduce_feats) if idx in pos]

        label = data.label_list[idx]
        points[label] = samples
    
    # for key in points.keys():
    #     print('key', key)
    #     print('point', points[key])

    # print(points)
    name = 'points_' + args.dataset
    all_dict[name] = points
    json_add(all_dict, draw_center_r_path)

def get_probs(args, data, outputs):
    
    interval = 0.1
    y_true, y_pred, y_prob = outputs[0], outputs[1], outputs[3]
    confidences = []
    score = 0
    while True:
        score += interval

        if score >= 1:
            break
        confidences.append(round(score, 2))

    print('confs', confidences)
    print(y_prob.shape)

    all_dicts = {}

    known_intents = []
    open_intents = []

    for conf in confidences:
        up_score = conf
        low_score = conf - interval
        pos = [idx for idx, prob in enumerate(probs) if (prob >= low_score and prob <= up_score)]
        num_knowns = len([p for p in pos if y_pred[p] != data.unseen_token_id])
        num_opens = -len([p for p in pos if y_pred[p] == data.unseen_token_id])

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


def save_detect_frontend_results(manager, args, data, outputs):

    y_true, y_pred, feat, y_prob = outputs[0], outputs[1], outputs[2], outputs[3]
    results_path = os.path.join(args.train_data_dir, args.type, 'results.csv')
    
    static_dir = os.path.join(args.frontend_dir, args.type)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    #save true_false predictions
    predictions = list([data.label_list[idx] for idx in y_pred]) 
    true_labels = list([data.label_list[idx] for idx in y_true]) 
    predict_t_f, predict_t_f_fine = cal_true_false(true_labels, predictions)
    csv_to_json(results_path, static_dir)

    tf_overall_path = os.path.join(static_dir, 'true_false_overall.json')
    tf_fine_path = os.path.join(static_dir, 'true_false_fine.json')

    results = {}
    results_fine = {}
    key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.labeled_ratio) + '_' + str(args.method)
    if os.path.exists(tf_overall_path):
        results = json_read(tf_overall_path)

    results[key] = predict_t_f

    if os.path.exists(tf_fine_path):
        results_fine = json_read(tf_fine_path)
    results_fine[key] = predict_t_f_fine

    json_add(results, tf_overall_path)
    json_add(results_fine, tf_fine_path)

    # print('test_results', data_diagram)

def save_detect_table_results_to_frontend(args, data, outputs):

    test_trues = list([data.label_list[idx] for idx in outputs[1]]) 
    test_preds = list([data.label_list[idx] for idx in outputs[0]]) 
    
    test_texts = [example.text_a for example in data.test_examples]
    
    known_label_list = test_preds.copy()
    known_label_list_unique = list(dict.fromkeys(known_label_list))
    dataset_info = {}
    dataset_info_json_path = os.path.join(sys.path[0], '/static/jsons/open_intent_detection', 'analysis_table_info.json')
    if os.path.exists(dataset_info_json_path):
        dataset_info = json_read(dataset_info_json_path)
    class_list = []
    known_num = 0
    for known_label_item in known_label_list_unique:
        text_list = []
        text_true_list_tmp = []
        # 获取等于当前label的
        # known_label_item_ids = list([ np.where( known_label_item ==  test_preds)[0] ])
        known_label_item_ids = list([ i for i,x in enumerate(known_label_list) if x == known_label_item ])
        known_num += len(known_label_item_ids)
        # print('=='*20,'\n\n',known_label_item, '\t', len(known_label_item_ids))
        for i in known_label_item_ids :
            text_true_list_tmp.append(test_trues[i])
            text_list.append({
                "dataset_name":args.dataset, "class_type":'known',
                "label_name": test_trues[i],
                "method": args.method,
                "text": test_texts[i]
            })
        class_list.append({"label_name": known_label_item, "label_text_num":len(known_label_item_ids),
            "dataset_name":args.dataset, "method": args.method, "class_type":'known'})
        # print('=='*20,'\n\n',known_label_item, '\t\n', text_true_list_tmp,'\n\n','=='*20)
        # add text_list to dataset_info
        dataset_info['text_list_'+args.dataset+"_"+args.method+"_known_"+known_label_item] = text_list
    # add class_list to dataset_info
    dataset_info["class_list_"+args.dataset+"_"+args.method+"_known"] = class_list
    # save dataset_info to frontend_file
    json_add(dataset_info, dataset_info_json_path)
    # with open(dataset_info_json_path, 'w') as write_f:
    #     json.dump(dataset_info, write_f, indent=4)