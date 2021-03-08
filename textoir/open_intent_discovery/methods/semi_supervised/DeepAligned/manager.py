from open_intent_discovery.utils import *
from .pretrain import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):
        
        #Save models and trainined data
        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
        output_file_name = "_".join([str(x) for x in concat_names])
        output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
        self.output_file_dir = os.path.join(output_dir, args.save_results_path)
        self.model_dir = os.path.join(output_dir, args.model_dir)
        self.pretrain_model_dir = os.path.join(output_dir, 'pretrain')

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id    

        Model = Backbone.__dict__[args.backbone]
        
        if args.train:
            
            self.best_eval_score = 0
            self.centroids = None
            pretrained_model = self.pre_train(args, data) 
        
            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data, pretrained_model) 
            else:
                self.num_labels = data.num_labels  
            
            self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)
            self.load_pretrained_model(args, pretrained_model)

        else:
            
            pretrained_model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.n_known_cls)
            pretrained_model = self.restore_model(pretrained_model, self.pretrain_model_dir)

            if args.cluster_num_factor > 1:
                self.num_labels = self.predict_k(args, data, pretrained_model) 
            else:
                self.num_labels = data.num_labels 

            self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)
            self.model = self.restore_model(self.model, self.model_dir)
        

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)   

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args)

        self.test_results, self.predictions, self.true_labels = None, None, None

    def pre_train(self, args, data):
        
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pretraining finished...')

        if args.save:
            self.save_model(manager_p.model, self.pretrain_model_dir)

        return manager_p.model

    def get_features_labels(self, dataloader, model, args):
        
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext = True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data, pretrained_model):
        print('Predict K begin...')

        feats, _ = self.get_features_labels(data.train_dataloader, pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('Predict K finished.. K={}, mean_cluster_size={}'.format(num_labels, drop_out))
        return num_labels
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer

    def evaluation(self, args, data, show=True):

        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        km = KMeans(n_clusters = self.num_labels).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        
        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]:i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])
        
        self.predictions = list([data.all_label_list[idx] for idx in y_pred])
        self.true_labels = list([data.all_label_list[idx] for idx in y_true])
        
        cm = confusion_matrix(y_true,y_pred) 
        
        if show:
            print('results',results)
            print('y_pred',y_pred)
            print('y_true',y_true)
            print('confusion matrix',cm)

        self.test_results = results

    def alignment(self, km, args):

        if self.centroids is not None:

            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]
                
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
        
        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.input_ids, data.input_mask, data.segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data): 

        best_model = None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            feats, _ = self.get_features_labels(data.train_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            
            eval_score = metrics.silhouette_score(feats, km.labels_)
            print('eval_score',eval_score)

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break 
            
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(train_dataloader, desc="Training(All)"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_fct = nn.CrossEntropyLoss()
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, loss_fct = loss_fct, mode='train')
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss',tr_loss)
        
        self.model = best_model

        if args.save:
            self.save_model(self.model, self.model_dir)
            

    def load_pretrained_model(self, args, pretrained_model):
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, model, model_dir):
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))

        return model


    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_model(self, model, model_dir):
            
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        save_model = model.module if hasattr(model, 'module') else model  
        model_file = os.path.join(model_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(model_dir, CONFIG_NAME)
        torch.save(save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(save_model.config.to_json_string())  

    def save_results(self, args, data):
        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        #save known intents
        np.save(os.path.join(self.output_file_dir, 'labels.npy'), data.all_label_list)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
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
        
        static_dir = os.path.join(args.frontend_dir, args.type)
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        #save true_false predictions
        predict_t_f, predict_t_f_fine = cal_true_false(self.true_labels, self.predictions)
        csv_to_json(results_path, static_dir)

        tf_overall_path = os.path.join(static_dir, 'ture_false_overall.json')
        tf_fine_path = os.path.join(static_dir, 'ture_false_fine.json')

        results = {}
        results_fine = {}
        key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.cluster_num_factor) + '_' + str(args.method)
        if os.path.exists(tf_overall_path):
            results = json_read(tf_overall_path)

        results[key] = predict_t_f

        if os.path.exists(tf_fine_path):
            results_fine = json_read(tf_fine_path)
        results_fine[key] = predict_t_f_fine

        json_add(results, tf_overall_path)
        json_add(results_fine, tf_fine_path)

        print('test_results', data_diagram)
        
    
