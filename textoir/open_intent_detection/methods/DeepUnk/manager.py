from open_intent_detection.utils import *
from open_intent_detection.pretrain import *
from .loss import AMSoftmax
import open_intent_detection.Backbone as Backbone
from sklearn.neighbors import LocalOutlierFactor

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

        
class ModelManager:
    
    def __init__(self, args, data):
        
        Model = Backbone.__dict__[args.backbone]
        self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
        
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_train_optimization_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args)
        
        self.loss_fct = None

        self.best_features = None
        self.best_eval_score = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_feat_label(self, data, args, dataloader, eval=False):
        
        with torch.no_grad():
            self.model.classifier.weight.copy_(self.loss_fct.W.T)

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        eval_loss = 0
        nb_tr_steps = 0

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            with torch.set_grad_enabled(False):
                # loss_fct = AMSoftmax(n_classes = data.num_labels, in_feats = data.num_labels)
                features, logits = self.model(input_ids, segment_ids, input_mask)
                if eval:
                    loss = self.loss_fct(features, label_ids)
                    eval_loss += loss.item()
                    nb_tr_steps += 1

            total_features = torch.cat((total_features, features))
            total_labels = torch.cat((total_labels, label_ids))
            total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        features = total_features.detach().cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
    
        if eval:
            loss = eval_loss / nb_tr_steps
            return loss, y_true, y_pred, features

        else:
            return y_true, y_pred, features

    def classify_lof(self, args, data, preds, feat_train, feat_pred):
        
        lof = LocalOutlierFactor(contamination = 0.05,novelty=True, n_jobs=-1)
        lof.fit(feat_train)
        y_pred_lof = pd.Series(lof.predict(feat_pred))
        preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_token_id

        return preds

    def evaluation(self, args, data, mode="eval", show=False):
        
        if mode == 'train':

            y_true, y_pred, features = self.get_feat_label(data, args, data.train_dataloader) 
            train_acc = round(accuracy_score(y_true, y_pred) * 100, 2)

            return train_acc

        if mode == 'eval':
            # _, _, feat_train = self.get_feat_label(data, args, data.train_dataloader)
            # y_pred = self.classify_lof(args, data, y_pred, feat_train, feat_pred)

            eval_loss, y_true, y_pred, features = self.get_feat_label(data, args, data.eval_dataloader, eval=True) 
            # eval_acc = round(accuracy_score(y_true, y_pred) * 100, 2)

            return eval_loss

        elif mode == 'test':
            
            _, _, feat_train = self.get_feat_label(data, args, data.train_dataloader)
            y_true, y_pred, feat_pred = self.get_feat_label(data, args, data.test_dataloader)

            y_pred = self.classify_lof(args, data, y_pred, feat_train, feat_pred)

            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Acc'] = acc
            self.test_results = results

            if show:
                print('test_results',results)

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
                         t_total=self.num_train_optimization_steps)   
        return optimizer

    def train(self, args, data):     
        best_model = None
        wait = 0
        cur_loss = None
        
        self.loss_fct = AMSoftmax(n_classes = data.num_labels, in_feats = args.feat_dim)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, label_ids, feature_ext = True)
                    loss = self.loss_fct(features, label_ids)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data, mode='train')
            print('train_acc', eval_score)

            eval_loss = self.evaluation(args, data, mode="eval")
            print('eval_loss',eval_loss)
            
            if self.best_eval_score is None:
                self.best_eval_score = eval_score
                best_model = copy.deepcopy(self.model)
            else:    
                if eval_score >= self.best_eval_score:
                    best_model = copy.deepcopy(self.model)
                    self.best_eval_score = eval_score
            
            if cur_loss is None:
                cur_loss = eval_loss
            else:
                if eval_loss <= cur_loss:
                    cur_loss = eval_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= 5:
                        break
                        
            print('cur_wait', wait)

            self.evaluation(args, data, mode="test", show=True)
                
        self.model = best_model 
        self.save_model(args)
    
    def freeze_bert_parameters(self, model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    def save_results(self, args):

        method_dir = os.path.join(args.type, 'methods', args.method)
        args.save_results_path = os.path.join(method_dir, args.save_results_path)
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        #save centroids, delta_points

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        result_file = 'results.csv'
        results_path = os.path.join(args.save_results_path, result_file)
        
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

    def save_model(self, args):
        args.pretrain_dir = os.path.join(args.type, 'methods', args.method, args.pretrain_dir)
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())  




  

    
    
