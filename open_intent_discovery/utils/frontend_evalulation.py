from collections import defaultdict
import json
import os 
import numpy as np

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

def save_test_results(args, test_results):
    
    if os.path.exists(args.test_results_dir):
        json_data = json_read(args.test_results_dir)
    
    
    record_name =  str(args.dataset) + '_' + str(args.method) + '_' + str(args.log_id)
    json_data[record_name]={}
    json_data[record_name]['ACC'] = test_results['ACC']
    json_data[record_name]['NMI'] = test_results['NMI']
    json_data[record_name]['ARI'] = test_results['ARI']

    json_add(json_data, args.test_results_dir)

def save_evaluation_results(args, data, results):

    save_dir = os.path.join(args.frontend_result_dir, args.type)  
    ######
    ##alignment
    from utils.metrics import hungray_aligment
    ind, w = hungray_aligment(results['y_true'], results['y_pred'])
    d_ind = {i[0]: i[1] for i in ind}
    import pandas as pd
    results['y_pred'] = pd.Series(results['y_pred']).map(d_ind)

    """
    for i, j in ind:
        pred_label = results['y_pred']
    """
    ######
    predictions = list([data.all_label_list[idx] for idx in results['y_pred']])
    true_labels = list([data.all_label_list[idx] for idx in results['y_true']])

    predict_t_f, predict_t_f_fine = cal_true_false(true_labels, predictions)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tf_overall_path = os.path.join(save_dir, 'true_false_overall.json')
    tf_fine_path = os.path.join(save_dir, 'true_false_fine.json')

    if not os.path.exists(tf_overall_path):
        f = open(tf_overall_path, 'w')
    elif os.path.exists(tf_overall_path) and (os.path.getsize(tf_overall_path) != 0):
        results = json_read(tf_overall_path)

    if not os.path.exists(tf_fine_path):
        f = open(tf_fine_path, 'w')
    elif os.path.exists(tf_fine_path) and (os.path.getsize(tf_fine_path) != 0):
        results_fine = json_read(tf_fine_path)

    results = {}
    results_fine = {}
    if os.path.exists(tf_overall_path):
        results = json_read(tf_overall_path)
    if os.path.exists(tf_fine_path):
        results_fine = json_read(tf_fine_path)
    key = str(args.dataset) + '_'  + str(args.method) + '_' + str(args.log_id)

    
    results[key] = predict_t_f

    results_fine[key] = predict_t_f_fine

    json_add(results, tf_overall_path)
    json_add(results_fine, tf_fine_path)  






        








