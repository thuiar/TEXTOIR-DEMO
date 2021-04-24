import open_intent_detection.Backbone as Backbone
from open_intent_detection.utils import *
from .loss import *
from sklearn.neighbors import LocalOutlierFactor


class ModelManager:
    
    def __init__(self, args, data):


        Model = Backbone.__dict__[args.backbone]
        self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels, is_doc = True)
        
        if args.freeze_bert_parameters:
            freeze_bert_parameters(self.model)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
        self.num_train_optimization_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args)

        
        self.test_results, self.predictions,  self.true_labels = None, None, None
        
        self.model_dir, self.output_file_dir = set_path(args)

        self.loss_fct = AMSoftmaxORI(in_feats = args.feat_dim, n_classes = data.num_labels)

        if args.train_detect:

            self.best_eval_score = 0
            self.best_features = None

        else:
            
            self.model = restore_model(self.model, self.model_dir)
            self.loss_fct = restore_loss(self.loss_fct, self.model_dir)
            self.best_features = np.load(os.path.join(self.output_file_dir, 'features.npy'))


    def classify_lof(self, args, data, preds, feat_train, feat_pred):
        
        lof = LocalOutlierFactor(contamination = 0.05, novelty=True, n_jobs=-1)
        lof.fit(feat_train)
        y_pred_lof = pd.Series(lof.predict(feat_pred))
        preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_token_id

        return preds

    def get_feat_label(self, data, args, dataloader):

        
        with torch.no_grad():
            self.model.classifier.weight.copy_(self.loss_fct.W.T)

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            with torch.set_grad_enabled(False):
                features, logits = self.model(input_ids, segment_ids, input_mask)

            total_features = torch.cat((total_features, features))
            total_labels = torch.cat((total_labels, label_ids))
            total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        features = total_features.detach().cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
    
        return y_true, y_pred, features

    def evaluation(self, args, data, dataloader, show=False):
    
        y_true, y_pred, feat = self.get_feat_label(data, args, dataloader)

        y_pred = self.classify_lof(args, data, y_pred, self.best_features, feat)
        
        cm = confusion_matrix(y_true,y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc
        self.test_results = results

        if show:
            print('test_results',results)

        return (y_pred, y_true, feat)

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
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', loss_fct = self.loss_fct)
                    # loss = self.loss_fct(features, label_ids)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            y_true, y_pred, _ = self.get_feat_label(data, args, data.eval_dataloader) 
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            print('eval_score', eval_score)
            
            if eval_score > self.best_eval_score:
                
                _, _, features = self.get_feat_label(data, args, data.train_dataloader)
                self.best_features = features
                self.best_eval_score = eval_score
                best_model = copy.deepcopy(self.model)
                wait = 0

            elif self.best_eval_score>0:
                wait += 1
                if wait >= args.wait_patient:
                    break
                        
            print('cur_wait', wait)

            # self.evaluation(args, data, show=True)
                
        self.model = best_model 

        np.save(os.path.join(self.output_file_dir, 'features.npy'), self.best_features)
        
        if args.save_detect:
            save_model(self.model, self.model_dir)
            save_loss(self.loss_fct, self.model_dir)
  

    
    
