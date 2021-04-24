from open_intent_detection.utils import *

def get_manager(args, data, mode):

    if not os.path.exists(os.path.join(args.train_data_dir, args.type)):
        os.makedirs(os.path.join(args.train_data_dir, args.type))

    if mode == 'detection':
        module_names = ['open_intent_detection', 'methods', args.method, 'manager']
    elif mode == 'discovery':
        module_names = ['open_intent_discovery', 'methods', args.setting, args.method, 'manager']
    
    import_name = ".".join(module_names)
    method = importlib.import_module(import_name) 
    manager = method.ModelManager(args, data)   

    return manager


def save_pipe(save_path, file_name, texts, labels):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['text', 'label'])
        for text, label in zip(texts, labels):
            writer.writerow([text, label])     

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)
    
def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path, allow_pickle=True)
    return npy_file

def save_pipeline_backend_results(all_results, args, data):

    var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
    names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(all_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    result_file = '{}_results.csv'.format('-'.join([args.detect_method, args.setting]))
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

def predict_save_detection_results(args, data, manager):
    
    #Labeled Train Data
    labeled_labels = [example.label for example in data.train_examples]
    labeled_texts = [example.text_a for example in data.train_examples]
    print('num_labeled_samples', len(labeled_labels))
    #Unlabeled Train Data 
    outputs = manager.evaluation(args, data, data.unlabeled_dataloader)
    unlabeled_texts = [example.text_a for example in data.unlabeled_examples]
    unlabeled_labels = list([data.all_label_list[idx] for idx in outputs[0]])
    print('num_unlabeled_samples', len(unlabeled_labels))

    train_labels = labeled_labels + unlabeled_labels
    train_texts = labeled_texts + unlabeled_texts
    save_pipe(args.pipe_results_path, 'train.tsv', train_texts, train_labels)
    #Eval Data
    eval_labels = [example.label for example in data.eval_examples]
    eval_texts = [example.text_a for example in data.eval_examples]
    save_pipe(args.pipe_results_path, 'dev.tsv',  eval_texts, eval_labels)
    #Test Data
    outputs = manager.evaluation(args, data, data.test_dataloader, show=True)
    test_trues = list([data.all_label_list[idx] for idx in outputs[1]]) 
    test_preds = list([data.label_list[idx] for idx in outputs[0]]) 
    test_texts = [example.text_a for example in data.test_examples]

    #Save Data
    if args.save_detect:
        save_pipe(args.pipe_results_path, 'detection.tsv', test_texts, test_preds)
        save_pipe(args.pipe_results_path, 'test.tsv', test_texts, test_trues)
        save_npy(np.array(data.all_label_list), args.pipe_results_path, 'all_labels.npy')
        save_npy(np.array(data.known_label_list), args.pipe_results_path, 'known_labels.npy')

    # Save Data to frontend

    ids_known = [idx for idx, label in enumerate(outputs[0]) if label != data.unseen_token_id]
    known_label_list = list([ example.label for i, example in enumerate(data.train_examples) if i in ids_known ])
    known_label_list = list(dict.fromkeys(known_label_list))
    # print('=='*20,'\n\ntrue:\t',test_trues, '\t\npred:\t', test_preds,'\n\n','=='*20)
    # detected_known_label_list = list([ data.all_label_list[idx] for idx in outputs[1] if idx >= args.unseen_token_id ])
    
    dataset_info = {}
    dataset_info_json_path = os.path.join(args.root_path, 'frontend/static/jsons/data_annotation', 'dataset_info.json')
    if os.path.exists(dataset_info_json_path):
        dataset_info = json_read(dataset_info_json_path)
        # with open(dataset_info_json_path, 'r') as load_f:
        #     dataset_info = json.load(load_f)
    # # known_label_list = list(dict.fromkeys(test_trues))
    # known_label_list = list(dict.fromkeys(known_label_list))
    class_list = []
    known_num = 0
    for known_label_item in known_label_list:
        text_list = []
        text_true_list_tmp = []
        # 获取等于当前label的
        known_label_item_ids = list([ i for i,x in enumerate(test_preds) if x == known_label_item ])
        known_num += len(known_label_item_ids)
        # print('=='*20,'\n\n',known_label_item, '\t', len(known_label_item_ids))
        for i in known_label_item_ids :
            text_true_list_tmp.append(test_trues[i])
            text_list.append({
                "dataset_name":args.dataset, "class_type":'known',
                "label_name": test_trues[i],
                "can_1": "","can_2": "","can_3": "",
                "text": test_texts[i]
            })
        class_list.append({"label_name": known_label_item, "label_text_num":len(known_label_item_ids),
            "dataset_name":args.dataset, "class_type":'known'})
        # print('=='*20,'\n\n',known_label_item, '\t\n', text_true_list_tmp,'\n\n','=='*20)
        # add text_list to dataset_info
        dataset_info['text_list_'+args.dataset+"_known_"+known_label_item] = text_list
    # add class_list to dataset_info
    dataset_info["class_list_"+args.dataset+"_known"] = class_list
    # add dataset_list to dataset_info
    if dataset_info.__contains__('dataset_list') == False :
        dataset_info['dataset_list'] = [{"dataset_name":args.dataset, "known_num":known_num}]
    else:
        dataset_list = dataset_info['dataset_list']
        dataset_id_in_dataset_list = -1
        for i,dataset_item in enumerate(dataset_list):
            if dataset_item['dataset_name'] == args.dataset:
                dataset_id_in_dataset_list = i
        if dataset_id_in_dataset_list != -1 :   # 说明找到了对应的位置,先取出，再赋值
            dataset_item = dataset_list[dataset_id_in_dataset_list]
            dataset_item['known_num'] = known_num
            # dataset_list[dataset_id_in_dataset_list] = {"dataset_name":args.dataset, "known_num":known_num}
            dataset_list[dataset_id_in_dataset_list] = dataset_item
            dataset_info['dataset_list'] = dataset_list
        else:
            dataset_info['dataset_list'].append({"dataset_name":args.dataset, "known_num":known_num})
    # save dataset_info to frontend_file
    with open(dataset_info_json_path, 'w') as write_f:
        json.dump(dataset_info, write_f, indent=4)
    
    return outputs[0]

def keyword_extraction(args, data, test_labels, ids):
    
    test_examples = np.array([example.text_a for example in data.test_examples])
    open_examples = test_examples[ids]

    keywords_model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = []
    # print('pipe.py-38:'+'\n'*3, ids)
    # Save Data to frontend
    # get all open label list
    open_intent_label_list = test_labels[ids]
    # 获取去重的label list
    open_intent_label_list_unique = np.unique(open_intent_label_list)
    # print('pipe.py-43:'+'\n'*3, open_intent_label_list_unique)
    dataset_info = {}
    dataset_info_json_path = os.path.join(sys.path[0], '../frontend/static/jsons/data_annotation', 'dataset_info.json')
    if os.path.exists(dataset_info_json_path):
        with open(dataset_info_json_path, 'r') as load_f:
            dataset_info = json.load(load_f)
    class_list = []
    unknown_num = 0
    conf_list_tmp = []
    i = 0
    for open_intent_label_item in open_intent_label_list_unique:
        pos = list( np.where(open_intent_label_list == open_intent_label_item)[0] )
        cluster_texts = open_examples[pos]
        # print('pipe.py-56:'+'\n'*3, pos, '\t\n', cluster_texts)
        unknown_num += len(cluster_texts)
        doc = " ".join(cluster_texts)
        # keywords_cluster = keywords_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n = 5)
        keywords_cluster = keywords_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), top_n = 3)
        strs_class_name = []
        for keyword_item in keywords_cluster:
            strs_class_name_tmp = '(' + str(keyword_item[0]) + ', ' + str( '%.2f' % (keyword_item[1]*100) ) + '%)'
            strs_class_name.append(strs_class_name_tmp)
        class_item = ", ".join(strs_class_name)
        # print('pipe.py-63:'+'\n'*3, class_item)
        class_list.append({"label_name": class_item, "label_text_num":len(cluster_texts),
            "dataset_name":args.dataset, "class_type":'open'})
        text_list = []
        for sent in cluster_texts:
            keywords_sent = keywords_model.extract_keywords(sent, keyphrase_ngram_range=(1,2), top_n = 3)
            keywords_sent_len = len(keywords_sent)
            can_1,can_2,can_3 = 'None', 'None', 'None'
            conf_1,conf_2,conf_3 = '0', '0', '0'
            try:
                if keywords_sent_len == 0:
                    can_1 = keywords_cluster[0][0]
                    conf_1 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                    can_2 = keywords_cluster[1][0]
                    conf_2 = '%.2f' % (keywords_cluster[1][1]*100) + '%'
                    can_3 = keywords_cluster[2][0]
                    conf_3 = '%.2f' % (keywords_cluster[2][1]*100) + '%'
                elif keywords_sent_len == 1:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_cluster[0][0]
                    conf_2 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                    can_3 = keywords_cluster[1][0]
                    conf_3 = '%.2f' % (keywords_cluster[1][1]*100) + '%'
                elif keywords_sent_len == 2:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_sent[1][0]
                    conf_2 = '%.2f' % (keywords_sent[1][1]*100) + '%'
                    can_3 = keywords_cluster[0][0]
                    conf_3 = '%.2f' % (keywords_cluster[0][1]*100) + '%'
                elif keywords_sent_len == 3:
                    can_1 = keywords_sent[0][0]
                    conf_1 = '%.2f' % (keywords_sent[0][1]*100) + '%'
                    can_2 = keywords_sent[1][0]
                    conf_2 = '%.2f' % (keywords_sent[1][1]*100) + '%'
                    can_3 = keywords_sent[2][0]
                    conf_3 = '%.2f' % (keywords_sent[2][1]*100) + '%'
            except :
                print('pipe.py:\t104:\tthere has an error')

            text_list.append({"dataset_name":args.dataset, "class_type":'open',
                "label_name": class_item,
                "can_1": can_1,"can_2": can_2,"can_3":can_3,
                "conf_1": conf_1,"conf_2": conf_2,"conf_3": conf_3,
                "text": sent
            })
        dataset_info['text_list_'+args.dataset+"_open_"+class_item] = text_list
        conf_list_tmp.append((i, keywords_cluster[0][1]))
        i = i + 1
    conf_list_tmp.sort(key = operator.itemgetter(1), reverse=True)
    class_list_tmp = class_list
    print('119:\t', conf_list_tmp)
    new_class_list = []
    for idx in range(len(conf_list_tmp)):
        new_class_list.append( class_list[ conf_list_tmp[idx][0] ] )
        print('123:\t', class_list[ conf_list_tmp[idx][0] ])
    class_list = new_class_list
    
    print('126:\t', class_list)
    dataset_info["class_list_"+args.dataset+"_open"] = class_list
    dataset_list = dataset_info['dataset_list']
    dataset_id_in_dataset_list = -1
    for i,dataset_item in enumerate(dataset_list):
        if dataset_item['dataset_name'] == args.dataset:
            dataset_id_in_dataset_list = i
    if dataset_id_in_dataset_list != -1 :
        dataset_list[dataset_id_in_dataset_list]['unknown'] = unknown_num
        dataset_list[dataset_id_in_dataset_list]['open'] = len(open_intent_label_list_unique)
        dataset_info['dataset_list'] = dataset_list
    with open(dataset_info_json_path, 'w') as write_f:
        json.dump(dataset_info, write_f, indent=4)


def transform_label(ids, list_a, list_b, is_a=True):
    if is_a:
        str_label = [list_a[i] for i in ids]
        list_b2id = {k:v for v,k in enumerate(list_b)}
        trans_label = [list_b2id[s] for s in str_label]
    else:
        str_label = [list_b[i] for i in ids]
        list_a2id = {k:v for v,k in enumerate(list_a)}
        trans_label = [list_a2id[s] for s in str_label]
    return trans_label