import os
import json
import numpy as np
from sklearn.manifold import TSNE

def json_read(path):
    
    with open(path, 'r')  as f:
        json_r = json.load(f)

    return json_r

def json_add(predict_t_f, path):
    
    with open(path, 'w') as f:
        json.dump(predict_t_f, f, indent=4)

def save_analysis_table_results(args, data, results, pipeline = False, save_dir = 'open_intent_detection'):
        
    test_trues = list([data.label_list[idx] for idx in results['y_true']]) 
    test_preds = list([data.label_list[idx] for idx in results['y_pred']]) 
    
    test_texts = [example.text_a for example in data.dataloader.test_examples]

    save_dir = os.path.join(args.frontend_result_dir, save_dir) 
    
    if pipeline:
        save_file_name = args.exp_name + '.json'
    else:
        save_file_name = 'analysis_table_info.json' 
    
    results_path = os.path.join(save_dir, save_file_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(results_path):
        f = open(results_path, 'w')
    elif os.path.exists(results_path) and (os.path.getsize(results_path) != 0):
        results = json_read(results_path)
    
    predict_labels = np.unique(np.array(test_preds))
    known_samples_nums = 0
    class_list = []
    data_info = {}

    for label in predict_labels:
        
        if pipeline and (label == '<UNK>'):
            continue

        text_list = []
        text_true_list_tmp = []
        known_sample_ids = [idx for idx, elem in enumerate(test_preds) if elem == label]
        known_samples_nums += len(known_sample_ids)

        for idx in known_sample_ids:
            text_true_list_tmp.append(test_trues[idx])
            text_list.append(
                {
                    'dataset_name':args.dataset,
                    'class_type': 'known',
                    'label_name': test_trues[idx], 
                    'method': args.method,
                    'text': test_texts[idx]
                }
            )
        
        class_list.append(
            {
                'label_name': label,
                'label_text_num': len(known_sample_ids),
                'dataset_name': args.dataset, 
                'method': args.method,
                'class_type': 'known'
            }
        )

        if pipeline:
            text_sample_name = 'text_list_' + args.dataset + "_known_"  + label
        else:
            text_sample_name = 'text_list_' + args.dataset + "_" + args.method + "_" + args.log_id + "_known_"  + label

        data_info[text_sample_name] = text_list
    
    if pipeline:

        dataset_list = {}
        dataset_list["dataset_name"] = args.dataset
        dataset_list["known_num"] = len([elem for elem in test_preds if elem != '<UNK>'])
        dataset_list["unknown"] = len(test_trues) - dataset_list["known_num"]
        known_intent_num = len(np.unique(np.array(test_preds))) - 1
        dataset_list["open"] = len(data.all_label_list) - known_intent_num
        data_info["dataset_list"] = dataset_list
        
        class_sample_name = 'class_list_' + args.dataset + "_known" 
        
    else:
        class_sample_name = 'class_list_' + args.dataset + "_" + args.method + "_" + args.log_id + "_known" 

    data_info[class_sample_name] = class_list

    json_add(data_info, results_path)

def TSNE_reduce_feats(feats, dim):

    estimator = TSNE(n_components=dim)
    reduce_feats = estimator.fit_transform(feats)
    
    return reduce_feats

def save_point_results(args, data, results):
    
    test_feats = results['y_feat']
    reduce_feats = TSNE_reduce_feats(test_feats, 2)

    all_data = {} 
    if os.path.exists(args.analysis_output_dir):
        all_data = json_read(args.analysis_output_dir)

    data_points = {}
    points = {}
    reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2) ] for item in reduce_feats ]
    for idx in range(args.num_labels):
        pos = list(np.where(results['y_pred'] == idx)[0])
        label_item = data.label_list[idx]

        samples = []
        for i, feat in enumerate(reduce_feats):
            if i in pos:
                samples.append(feat)
        points[label_item] = samples
    data_points['points'] = points

    sample_name = args.dataset + '_' + args.method + '_' + args.log_id
    all_data[sample_name] = data_points 
    json_add(all_data, args.analysis_output_dir)

def save_MSP_results(args, data, results):
    
    interval = 0.1
    y_true, y_pred, y_prob = results["y_true"], results["y_pred"], results["y_prob"]

    xais = []
    score = 0

    while True:

        score += interval

        if score >= 1:
            break

        xais.append(round(score, 2))

    known_intents = []
    open_intents = []
    scores = []

    for point in xais:

        up_score = point
        low_score = point - interval

        pos = [idx for idx, prob in enumerate(y_prob) if (prob >= low_score and prob <= up_score)]

        num_knowns = len([p for p in pos if y_true[p] != data.unseen_label_id])
        num_opens = -len([p for p in pos if y_true[p] == data.unseen_label_id])

        known_intents.append(num_knowns)
        open_intents.append(num_opens)

        pos_neg = [idx for idx, prob in enumerate(y_prob) if prob <= point]
        pos_pos = [idx for idx, prob in enumerate(y_prob) if prob > point]

        neg_correct_samples = 0
        pos_correct_samples = 0
        for idx in pos_neg:

            if y_true[idx] == data.unseen_label_id:
                neg_correct_samples += 1

        for idx in pos_pos:
            if y_true[idx] == y_pred[idx]:
                pos_correct_samples += 1

        score = (neg_correct_samples + pos_correct_samples) / len(y_true)
        scores.append(score)

    all_data = {} 
    if os.path.exists(args.analysis_output_dir):
        all_data = json_read(args.analysis_output_dir)

    sample_data = {}
    sample_data["Known_Intent"] = known_intents
    sample_data["Open_Intent"] = open_intents
    sample = {}
    sample["data"] = sample_data

    data_x = {}
    data_x['xaxis'] = xais
    data_x["score"] = scores

    sample["data_x"] = data_x
    sample_name = args.dataset + '_' + args.method + '_' + args.log_id

    all_data[sample_name] = sample   

    json_add(all_data, args.analysis_output_dir)

def save_DOC_results(args, data, results):

    interval = 0.1
    y_true, y_pred, y_prob = results["y_true"], results["y_pred"], results["y_prob"]
    thresholds = results["thresholds"]

    xais_list = []
    score = 0

    while True:

        score += interval

        if score >= 1:
            break

        xais_list.append(round(score, 2))

    labels = np.unique(y_true)
    labels = labels[:-1]

    xais = {}
    labels_name = list([data.label_list[idx] for idx in labels]) 

    for label in labels:
        xais_name = str(labels_name[label]) + '_xaxis'
        xais[xais_name] = xais_list

    index = {}
    label_list = []
    threshold_list = []
    for key in thresholds.keys():
        threshold = thresholds[key]
        label_list.append(key)
        threshold_list.append(threshold)

    index["Intent_Class"] = label_list
    index["Intent_Threshold"] = threshold_list

    content = {}
    scores = []

    for i, label in enumerate(labels):
        
        known_intents = []
        open_intents = []

        label_pos = list(np.where(y_pred == label)[0])

        for point in xais_list:

            up_score = point
            low_score = point - interval

            pos = [idx for idx in label_pos if (y_prob[idx] >= low_score) and (y_prob[idx] <= up_score)]
            
            num_knowns = len([p for p in pos if y_true[p] != data.unseen_label_id])
            num_opens = -len([p for p in pos if y_true[p] == data.unseen_label_id])

            known_intents.append(num_knowns)
            open_intents.append(num_opens)
        
        intent_sample = {}
        intent_sample['Known_Intent'] = known_intents
        intent_sample['Open_Intent'] = open_intents
        
        label_name = labels_name[label]
        content[label_name] = intent_sample


        pos_neg = [idx for idx, prob in enumerate(y_prob) if prob <= threshold_list[i]]
        pos_pos = [idx for idx, prob in enumerate(y_prob) if prob > threshold_list[i]]

        neg_correct_samples = 0
        pos_correct_samples = 0
        for idx in pos_neg:

            if y_true[idx] == data.unseen_label_id:
                neg_correct_samples += 1

        for idx in pos_pos:
            if y_true[idx] == y_pred[idx]:
                pos_correct_samples += 1

        score = (neg_correct_samples + pos_correct_samples) / len(y_true)
        scores.append(score)

    index["Intent_score"] = scores

    all_data = {} 
    if os.path.exists(args.analysis_output_dir):
        all_data = json_read(args.analysis_output_dir)

    sample = {}
    sample["index"] = index
    sample["xaxis"] = xais
    sample["content"] = content
    sample_name = args.dataset + '_' + args.method + '_' + args.log_id

    all_data[sample_name] = sample
    
    json_add(all_data, args.analysis_output_dir)

def save_OpenMax_results(args, data, results):
    
    interval = 0.1
    y_true, y_pred, y_prob = results["y_true"], results["y_pred"], results["y_prob"]
    openmax_pred = results["openmax_pred"]
    softmax_pred = results["softmax_pred"]

    xais = []
    threshold = 0

    while True:

        threshold += interval

        if threshold >= 1:
            break

        xais.append(round(threshold, 2))

    known_intents = []
    open_intents = []
    predicted_open_intents = []
    scores = []

    for point in xais:

        up_score = point
        low_score = point - interval

        pos = [idx for idx, prob in enumerate(y_prob) if (prob >= low_score and prob <= up_score)]

        num_knowns = len([p for p in pos if y_true[p] != data.unseen_label_id])
        num_opens = -len([p for p in pos if y_true[p] == data.unseen_label_id])
        num_predicted_opens = -len([p for p in pos if openmax_pred == data.unseen_label_id])

        known_intents.append(num_knowns)
        open_intents.append(num_opens)
        predicted_open_intents.append(num_predicted_opens)

        pos_neg = [idx for idx, prob in enumerate(y_prob) if prob <= point]
        pos_pos = [idx for idx, prob in enumerate(y_prob) if prob > point]

        neg_correct_samples = 0
        pos_correct_samples = 0
        for idx in pos_neg:

            if (openmax_pred[idx] == data.unseen_label_id) or (y_true[idx] == data.unseen_label_id):
                neg_correct_samples += 1

        for idx in pos_pos:
            if y_true[idx] == softmax_pred[idx]:
                pos_correct_samples += 1

        score = (neg_correct_samples + pos_correct_samples) / len(y_true)
        scores.append(score)

    all_data = {} 
    if os.path.exists(args.analysis_output_dir):
        all_data = json_read(args.analysis_output_dir)

    index = {}
    index["score"] = scores
    index["xaxis"] = xais

    intent_data = {}
    intent_data["Known_Intent"] = known_intents
    intent_data["Open_Intent"] = open_intents


    sample_data = {}
    sample_data["data"] = intent_data
    sample_data["data_x"] = index
    sample_name = args.dataset + '_' + args.method + '_' + args.log_id
    all_data[sample_name] = sample_data


    json_add(all_data, args.analysis_output_dir)
