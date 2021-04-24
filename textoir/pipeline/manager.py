import os
import csv
import numpy as np

class PipelineManager:

    def __init__(self, predictions=None):
        self.predictions = predictions

    def discover(self, args, data, manager):
        
        print('Open Intent Discovery: Prediction begin...')
        new_intent_examples, known_intent_examples = [], []
        new_predictions, known_predictions = [], []
        new_texts, known_texts = [], []
        new_true_labels, known_true_labels = [], []

        for idx, example in enumerate(data.test_examples):
            if self.predictions[idx] == data.unseen_token:
                new_intent_examples.append(example)
                new_texts.append(example.text_a)
                new_true_labels.append(example.label)
            else:
                known_intent_examples.append(example)
                known_texts.append(example.text_a)
                known_predictions.append(self.predictions[idx])
                known_true_labels.append(example.label)

        data.test_dataloader = data.get_loader(new_intent_examples, args, data.all_label_list, 'test')
        # manager.num_labels = len(data.all_label_list) - len(data.known_label_list)
        manager.evaluation(args, data, show=False)

        texts = known_texts
        texts.extend(new_texts)
        
        self.predictions = known_predictions
        self.predictions.extend(manager.predictions)
        
        self.true_labels = known_true_labels
        self.true_labels.extend(new_true_labels)

        self.save_discover_results(args, texts, self.predictions, self.true_labels)    
        self.save_detect_results(args, data, texts, true_labels, typ[idx])

        print('Open Intent Prediction finished...')

    def evaluation(self, args, data, mode='detect'):
        npy_path = os.path.join(args.pipe_results_path, 'labels.npy')
        labels = np.load(npy_path)
        

        if mode == 'detect':
            labels = data.label_list
            label2id = {x:i for i,x in enumerate(labels)}
            for example in data.test_examples:
                if example.label not in data.known_label_list:
                    example.label = data.unseen_token

            y_true = np.array([label2id[x.label] for x in data.test_examples])
            y_pred = np.array([label2id[x] for x in self.predictions])

            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Acc'] = acc
            var = [args.dataset, args.type, args.detect_method, args.known_cls_ratio, args.labeled_ratio, args.seed]    
            names = ['dataset', 'mode', 'detect_method', 'known_cls_ratio', 'labeled_ratio', 'seed']
            
            file_name = 'results_detect.csv'

        elif mode == 'discover':
            labels = data.all_label_list
            label2id = {x:i for i,x in enumerate(labels)}
            y_true = np.array([label2id[x] for x in self.true_labels])
            y_pred = np.array([label2id[x] for x in self.predictions])

            results = clustering_score(y_true, y_pred)
            var = [args.dataset, args.type, args.discover_method, args.known_cls_ratio, args.labeled_ratio, args.seed]    
            names = ['dataset', 'mode', 'discover_method', 'known_cls_ratio', 'labeled_ratio', 'seed']
            file_name = 'results_discover.csv'

        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values()) 
        
        results_path = os.path.join(args.pipe_results_path, file_name)
        
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
        



    