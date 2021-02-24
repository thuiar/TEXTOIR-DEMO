from open_intent_discovery.utils import *
from open_intent_discovery.Backbone import BertForConstrainClustering as CDACPlus
from .pretrain import *

class ModelManager:
    
    def __init__(self, args, data):

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id    
        
        self.num_labels = data.num_labels   
        self.model = CDACPlus.from_pretrained(args.bert_model, cache_dir = "", num_labels = self.num_labels)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)  

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs

        self.best_eval_score = 0

        self.predictions = None
        self.test_results = None
        self.true_labels = None

    def initialize_centroids(self, args, data):

        print('Initialize centroids...')

        _, feats, _ = self.get_preds_feats(data.train_unlabeled_dataloader, self.model)
        km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)

        print('Initialization finished...')

        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)
    
    def get_preds_feats(self, dataloader, model):
        
        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_feats = torch.empty((0, self.num_labels)).to(self.device)
        total_qs = torch.empty((0, self.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                out1, out2 = model(input_ids, segment_ids, input_mask)
    
            total_labels = torch.cat((total_labels, label_ids))
            total_feats = torch.cat((total_feats, out1))
            total_qs = torch.cat((total_qs, out2))

        y_true = total_labels.cpu().numpy()
        feat = total_feats.cpu().numpy()
        q = total_qs.cpu().numpy()

        return y_true, feat, q
    
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

    def evaluation(self, data, args = None, show=True):

        y_true, _, test_q = self.get_preds_feats(data.test_dataloader, self.model)
        y_pred = test_q.argmax(1)
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

    def initialize_centroids(self, args, data):

        print('Initialize centroids...')

        _, feats, _ = self.get_preds_feats(data.train_dataloader, self.model)
        km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)

        print('Initialization finished...')

        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def train(self, args, data): 

        print('Pairwise-similarity Learning begin...')
        optimizer = self.get_optimizer(args)
        
        u = args.u
        l = args.l
        eta = 0

        eval_pred_last = np.zeros_like(data.eval_examples)
        
        results = {}
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Iteration (labeled)")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, u, l, 'train', label_ids)
                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad() 

            train_labeled_loss = tr_loss / nb_tr_steps
            print('train_labeled_loss', round(train_labeled_loss, 4))

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            for step, batch in enumerate(tqdm(data.train_semi_dataloader, desc="Iteration (all train)")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, u, l, 'train', label_ids, True)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
            
            train_loss = tr_loss / nb_tr_steps
            print('train_loss', round(train_loss, 4))

            eval_true, eval_logits, eval_q = self.get_preds_feats(data.eval_dataloader, self.model)
            eval_pred = eval_logits.argmax(1)
            eval_results = clustering_score(eval_true, eval_pred) 
            print('eval_results', eval_results)

            delta_label = np.sum(eval_pred != eval_pred_last).astype(np.float32) / eval_pred.shape[0]
            eval_pred_last = np.copy(eval_pred)

            results['u'] = u
            results['l'] = l
            results['train_labeled_loss'] = train_labeled_loss
            results['train_loss'] = train_loss
            results['delta_label'] = delta_label

            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
            print('loss={}, (u, l) ={}, {}'.format(train_loss, round(u, 4), round(l, 4)))
            
            eta += 1.1 * 0.009
            u = 0.95 - eta
            l = 0.455 + eta * 0.1
            if u < l:
                break

        print('Pairwise-similarity Learning finished!')

        self.initialize_centroids(args, data)

        best_model = None
        wait = 0
        wait, patient = 0, 5
        y_pred_last = None

        for epoch in range(args.num_train_epochs):
            
            #evaluation
            eval_true, _, eval_q = self.get_preds_feats(data.eval_dataloader, self.model)
            eval_pred = eval_q.argmax(1)
            results = clustering_score(eval_true, eval_pred) 
            print('eval_results', results)

            #early stop
            if results['NMI'] > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = results['NMI']
            else:
                wait += 1
                if wait > patient:
                    break

            self.model = best_model

            #converge
            _, _, q_all = self.get_preds_feats(data.train_dataloader, self.model)
            p_target = target_distribution(q_all)
            y_pred = q_all.argmax(1)

            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if epoch > 0 and delta_label < 0.001:
                print(epoch, delta_label, 'break')
                break
            
            # Fine-tuning with auxiliary distribution
            self.model.train()
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            for step, batch in enumerate(data.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits, q = self.model(input_ids, segment_ids, input_mask, mode='finetune')
                kl_loss = F.kl_div(q.log(), torch.Tensor(p_target[step * args.train_batch_size: (step + 1) * args.train_batch_size]).cuda())
                kl_loss.backward()

                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad() 
            
            train_loss = tr_loss / nb_tr_steps
            results['kl_loss'] = round(train_loss, 4)
            results['delta_label'] = delta_label.round(4)
            print('epoch:{}, results:{}'.format(epoch, results))


    def load_pretrained_model(self, args, pretrained_model):
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        method_dir = os.path.join(args.type, 'methods', args.setting, args.method)
        args.save_results_path = os.path.join(method_dir, args.save_results_path)
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'results'  + '.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
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
        

