import os
import json
import numpy as np
import logging
from sklearn.manifold import TSNE
from torch.utils.data import dataset
from keybert import KeyBERT

def json_read(path):
    
    with open(path, 'r')  as f:
        json_r = json.load(f)

    return json_r

def json_add(predict_t_f, path):

    with open(path, 'w') as f:
        json.dump(predict_t_f, f, indent=4)

def save_analysis_table_results(args, data, results, logger_name):
    
    test_trues = list([data.all_label_list[idx] for idx in results['y_true']]) 
    test_preds = list([data.all_label_list[idx] for idx in results['y_pred']]) 
    
    if args.backbone == 'bert':
        test_texts = np.array([example.text_a for example in data.dataloader.test_examples])
    elif args.setting == 'unsupervised':
        test_texts = np.array([text for text in data.dataloader.ori_test_data.text])
    
    save_dir = os.path.join(args.frontend_result_dir, args.type) 
    save_file_name = 'analysis_table_info.json' 
    results_path = os.path.join(save_dir, save_file_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(results_path):
        f = open(results_path, 'w')
    elif os.path.exists(results_path) and (os.path.getsize(results_path) != 0):
        results = json_read(results_path)
    
    logger = logging.getLogger(logger_name)
    logger.info('Loading KeyBERT model start...')
    keywords_model = KeyBERT('distilbert-base-nli-mean-tokens')
    logger.info('Loading KeyBERT model finished...')

    keywords = []
    label_list = []
    predict_labels = np.unique(np.array(test_preds))
    
    dataset_info = {}

    for label in predict_labels:
        label_sample_ids = [idx for idx, elem in enumerate(test_preds) if elem == label]
        label_texts = test_texts[label_sample_ids]
        doc = " ".join(label_texts)
        keywords = keywords_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n = 3)

        name_list = []
        for keyword in keywords:
            name = '(' + str(keyword[0]) + ', ' + str(keyword[1]) + str( '%.2f' % (keyword[1]*100) ) + '%)'
            name_list.append(name)
        
        label_item = ', '.join(name_list)
        label_list.append({
            "label_name": label_item,
            "label_text_num": len(label_texts),
            "dataset_name": args.dataset,
            "method": args.method,
            "class_type": "open"
        })

        text_list = []
        for sent in label_texts:

            keywords_sent = keywords_model.extract_keywords(sent, keyphrase_ngram_range=(1,2), top_n = 3)
            keywords_sent_len = len(keywords_sent)
            can_1, can_2, can_3 = 'None', 'None', 'None'
            conf_1, conf_2, conf_3 = '0', '0', '0'

            if keywords_sent_len == 0:
                
                can_1 = keywords[0][0]
                conf_1 = '%.2f' % (keywords[0][1] * 100) + '%'
                can_2 = keywords[1][0]
                conf_2 = '%.2f' % (keywords[1][1] *100) + '%'
            
            elif keywords_sent_len == 1:
                
                can_1 = keywords_sent[0][0]
                conf_1 = '%.2f' % (keywords_sent[0][1] * 100) + '%'
                can_2 = keywords[0][0]
                conf_2 = '%.2f' % (keywords[0][1] * 100) + '%'
                can_3 = keywords[1][0]
                conf_3 = '%.2f' % (keywords[1][1] * 100) + '%'
            
            elif keywords_sent_len == 2:
                
                can_1 = keywords_sent[0][0]
                conf_1 = '%.2f' % (keywords_sent[0][1] * 100) + '%'
                can_2 = keywords_sent[1][0]
                conf_2 = '%.2f' % (keywords_sent[1][1] * 100) + '%'
                can_3 = keywords[0][0]
                conf_3 = '%.2f' % (keywords[0][1] * 100) + '%'

            elif keywords_sent_len == 3:

                can_1 = keywords_sent[0][0]
                conf_1 = '%.2f' % (keywords_sent[0][1] * 100) + '%'
                can_2 = keywords_sent[1][0]
                conf_2 = '%.2f' % (keywords_sent[1][1]*100) + '%'
                can_3 = keywords_sent[2][0]
                conf_3 = '%.2f' % (keywords_sent[2][1] * 100) + '%'
            
            else:
                print('Error')
        
            text_list.append(
                {
                    "dataset_name": args.dataset, 
                    "class_type":'open',
                    "label_name": label_item,
                    "method": args.method,
                    "can_1": can_1,
                    "can_2": can_2,
                    "can_3":can_3,
                    "conf_1": conf_1,
                    "conf_2": conf_2,
                    "conf_3": conf_3,
                    "text": sent
                }
            )
        
        sample_name = 'text_list_'+ args.dataset + "_" + args.method + "_" + str(args.log_id) + "_open_" + label_item
        dataset_info[sample_name] = text_list
    
    class_name = "class_list_"  + args.dataset + "_" + args.method + "_" + str(args.log_id) + "_open"
    dataset_info[class_name] = label_list

    json_add(dataset_info, results_path)