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

def save_test_results(args, test_results):

    if os.path.exists(args.test_results_dir):
        json_data = json_read(args.test_results_dir)
    
    
    record_name =  str(args.dataset) + '_' + str(args.method) + '_' + str(args.log_id)
    json_data[record_name]={}
    json_data[record_name]['Acc'] = test_results['Acc']
    json_data[record_name]['F1'] = test_results['F1']
    json_data[record_name]['F1-known'] = test_results['F1-known']
    json_data[record_name]['F1-open'] = test_results['F1-open']  

    json_add(json_data, args.test_results_dir)

def save_train_results(args, result_list):
    
    results_path = args.train_results_dir

    train_loss_list = []
    valid_score_list = []

    num_intervals = 20
    interval = int(len(result_list) / num_intervals) + 1

    for i, elem in enumerate(result_list):
        
        if i % interval == 0:
            train_loss_list.append(elem['train_loss'])
            valid_score_list.append(elem['eval_score'])
    
    record_name = 'detection_' + str(args.dataset) + '_' + str(args.method) + '_' + str(args.log_id)

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:

        json_data = {}
        json_data[record_name] = {}
        
        json_data[record_name]['Training'] = train_loss_list[:20]
        json_data[record_name]['Validation'] = valid_score_list[:20]

    else:

        with open(results_path, 'r') as f:
            json_data = json.load(f)

        json_data[record_name] = {}
        json_data[record_name]['Training'] = train_loss_list[:20]
        json_data[record_name]['Validation'] = valid_score_list[:20]

    json_add(json_data, results_path)

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

def save_evaluation_results(args, data, results):

    save_dir = os.path.join(args.frontend_result_dir, args.type)  

    predictions = list([data.label_list[idx] for idx in results['y_pred']])
    true_labels = list([data.label_list[idx] for idx in results['y_true']])

    predict_t_f, predict_t_f_fine = cal_true_false(true_labels, predictions)

    tf_overall_path = os.path.join(save_dir, 'true_false_overall.json')
    tf_fine_path = os.path.join(save_dir, 'true_false_fine.json')

    results = {}
    results_fine = {}
    key = str(args.dataset) + '_'  + str(args.method) + '_' + str(args.log_id)
    if os.path.exists(tf_overall_path):
        results = json_read(tf_overall_path)
    
    results[key] = predict_t_f

    if os.path.exists(tf_fine_path):
        results_fine = json_read(tf_fine_path)

    results_fine[key] = predict_t_f_fine

    json_add(results, tf_overall_path)
    json_add(results_fine, tf_fine_path)  






        








