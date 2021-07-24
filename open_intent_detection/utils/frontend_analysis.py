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

def save_analysis_table_results(args, data, results):

    test_trues = list([data.label_list[idx] for idx in results['y_true']]) 
    test_preds = list([data.label_list[idx] for idx in results['y_pred']]) 
    
    test_texts = [example.text_a for example in data.dataloader.test_examples]

    save_dir = os.path.join(args.frontend_result_dir, args.type) 
    save_file_name = 'analysis_table_info.json' 
    results_path = os.path.join(save_dir, save_file_name)
    if os.path.exists(results_path):
        results = json_read(results_path)
    
    predict_labels = np.unique(np.array(test_preds))
    known_samples_nums = 0
    class_list = []
    data_info = {}

    for label in predict_labels:
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
        text_sample_name = 'text_list_' + args.dataset + "_" + args.method + "_" + args.log_id + "_known_"  + label
        data_info[text_sample_name] = text_list
    
    class_sample_name = 'class_list_' + args.dataset + "_" + args.method + "_" + args.log_id + "_known" 
    data_info[class_sample_name] = class_list

    json_add(data_info, results_path)

def TSNE_reduce_feats(feats, dim):

    estimator = TSNE(n_components=dim)
    reduce_feats = estimator.fit_transform(feats)
    
    return reduce_feats

def save_point_results(args, data, results):
    
    centroids = np.load(os.path.join(args.method_output_dir, 'centroids.npy'))
    deltas = np.load(os.path.join(args.method_output_dir, 'deltas.npy'))

    test_trues = list([data.label_list[idx] for idx in results['y_true']]) 
    test_preds = list([data.label_list[idx] for idx in results['y_pred']]) 
    test_feats = results['y_feat']

    reduce_centers = TSNE_reduce_feats(centroids, 2)
    reduce_centers = [np.round(x, 2) for x in reduce_centers]
    reduce_feats = TSNE_reduce_feats(test_feats, 2)

    save_dir = os.path.join(args.frontend_result_dir, args.type) 
    save_file_name = args.method + '_analysis.json'
    results_path = os.path.join(save_dir, save_file_name)

    data_points = {}
    points = {}
    reduce_feats = [[round(float(item[0]), 2), round(float(item[1]), 2) ] for item in reduce_feats ]
    for idx in range(len(reduce_centers)):
        pos = list(np.where(results['y_pred'] == idx)[0])
        label_item = data.label_list[idx]

        samples = []
        for i, feat in enumerate(reduce_feats):
            if i in pos:
                samples.append(feat)
        points[label_item] = samples
    data_points['points'] = points

    all_dict = {}
    sample_name = args.dataset + '_' + args.method + '_' + args.log_id
    all_dict[sample_name] = data_points 
    json_add(all_dict, results_path)







