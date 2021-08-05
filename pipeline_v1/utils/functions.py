import os
import numpy as np
import pandas as pd   
import csv
import logging

def save_data(save_path, file_name, texts, labels):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['text', 'label'])
        for text, label in zip(texts, labels):
            writer.writerow([text, label]) 

def save_pipeline_results(args, data, manager, test_results):
    
    if args.backbone == 'bert':
        
        #Labeled Train Data
        train_labeled_labels = [example.label for example in data.dataloader.train_labeled_examples]
        train_labeled_texts = [example.text_a for example in data.dataloader.train_labeled_examples]
        
        #Unlabeled Train Data
        manager.test_dataloader = data.dataloader.train_unlabeled_loader
        train_unlabeled_results = manager.test(args, data, show = False)
        
        train_unlabeled_texts = [example.text_a for example in data.dataloader.train_unlabeled_examples]
        train_unlabeled_labels = list([data.all_label_list[idx] for idx in train_unlabeled_results['y_pred']])

        train_labels = train_labeled_labels + train_unlabeled_labels
        train_texts = train_labeled_texts + train_unlabeled_texts
        
        #Eval Data
        eval_labels = [example.label for example in data.dataloader.eval_examples]
        eval_texts = [example.text_a for example in data.dataloader.eval_examples]
        
        #Test Data
        test_labels = [data.label_list[idx] for idx in test_results['y_pred']]
        manager.test_dataloader = data.test_true_loader 
        test_results = manager.test(args, data, show = False) 
                          
        test_true_labels = [data.all_label_list[idx] for idx in test_results['y_true']]
        test_texts = [example.text_a for example in data.dataloader.test_examples]
        
        #Save file for training discovery
        save_data(args.pipe_data_dir, 'train.tsv', train_texts, train_labels)
        save_data(args.pipe_data_dir, 'dev.tsv', eval_texts, eval_labels)
        save_data(args.pipe_data_dir, 'test.tsv', test_texts, test_true_labels)
        save_data(args.pipe_data_dir, 'test_preds.tsv', test_texts, test_labels)

        #Save labels
        np.save(os.path.join(args.pipe_data_dir, 'all_labels.npy'), np.array(data.all_label_list))
        np.save(os.path.join(args.pipe_data_dir, 'known_labels.npy'), np.array(data.known_label_list))

def save_results(args, test_results):

    if 'y_true' in test_results.keys():
        del test_results['y_true']
    if 'y_pred' in test_results.keys():
        del test_results['y_pred']
    if 'feats' in test_results.keys():
        del test_results['feats']

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    type_results_path = os.path.join(args.results_dir, args.type)
    if not os.path.exists(type_results_path):
        os.makedirs(type_results_path)

    var = [args.dataset, args.detection_method, args.discovery_method, \
        args.known_cls_ratio, args.labeled_ratio, args.seed]

    names = ['dataset', 'detection_method', 'discovery_method', 'known_cls_ratio', 'labeled_ratio',  'seed']

    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(type_results_path, args.results_file_name)
    
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
    
    logger = logging.getLogger(args.logger_name)
    logger.info('test_results: %s', data_diagram)

def combine_test_results(args,  detection_results, \
    discovery_data, discovery_results, logger):
    
    pred_known_ids = [idx for idx, label in enumerate(detection_results['y_pred']) if label != discovery_data.unseen_token_id]
    pred_open_ids = [idx for idx, label  in enumerate(detection_results['y_pred']) if label == discovery_data.unseen_token_id]

    open_feats = discovery_results['feats'][pred_open_ids]

    open_k_num = discovery_data.num_labels - discovery_data.n_known_cls
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=open_k_num, n_jobs=-1, random_state = args.seed)
    km.fit(open_feats)
    open_labels = km.labels_ + discovery_data.n_known_cls

    discovery_results['y_pred'][pred_known_ids] = detection_results['y_pred'][pred_known_ids]
    discovery_results['y_pred'][pred_open_ids] = open_labels

    test_known_ids =  [idx for idx, label in enumerate(discovery_results['y_true']) \
        if discovery_data.all_label_list[label] in discovery_data.known_label_list]

    from sklearn.metrics import accuracy_score, f1_score
    known_intent_acc = accuracy_score(discovery_results['y_true'][test_known_ids], discovery_results['y_pred'][test_known_ids])
    known_intent_f1 = f1_score(discovery_results['y_true'][test_known_ids], discovery_results['y_pred'][test_known_ids], average = 'macro')
    
    test_open_ids = [idx for idx, label in enumerate(discovery_results['y_true']) \
        if discovery_data.all_label_list[label] not in discovery_data.known_label_list]
    print(discovery_results)
    
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    open_intent_nmi = normalized_mutual_info_score(discovery_results['y_true'][test_open_ids], discovery_results['y_pred'][test_open_ids])
    open_intent_ari = adjusted_rand_score(discovery_results['y_true'][test_open_ids], discovery_results['y_pred'][test_open_ids])

    results = {
                'known_intent_acc': round(known_intent_acc * 100, 2),
                'known_intent_f1': round(known_intent_f1 * 100, 2),
                'open_intent_nmi': round(open_intent_nmi * 100, 2),
                'open_intent_ari': round(open_intent_ari * 100, 2),
              }

    logger.info("***** Final results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))  

    return results 