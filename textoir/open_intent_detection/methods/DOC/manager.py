from open_intent_detection.utils import *
from open_intent_detection.pretrain import *
import open_intent_detection.Backbone as Backbone
from scipy.stats import norm as dist_model

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
        
        self.test_results, self.predictions,  self.true_labels = None, None, None
        
        #Save models and trainined data
        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
        output_file_name = "_".join([str(x) for x in concat_names])
        output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
        self.output_file_dir = os.path.join(output_dir, args.save_results_path)
        self.model_dir = os.path.join(output_dir, args.model_dir)

        if args.train_detect:

            self.best_eval_score = 0
            self.best_mu_stds = None

        else:

            self.restore_model(args)
            # self.best_mu_stds = np.load(os.path.join(self.output_file_dir, 'mu_stds.npy'))
            try:
                self.best_mu_stds = np.load(os.path.join(self.output_file_dir, 'mu_stds.npy'))
            except:
                self.best_mu_stds = self.get_mu_stds(args, data)
        

    def classify_doc(self, data, args, probs, mu_stds):

        thresholds = {}
        for col in range(data.num_labels):
            threshold = max(0.5, 1 - args.scale * mu_stds[col][1])
            label = data.known_label_list[col]
            thresholds[label] = threshold

        print('DOC_thresholds', thresholds)
        
        preds = []
        for p in probs:
            max_class = np.argmax(p)
            max_value = np.max(p)
            threshold = max(0.5, 1 - args.scale * mu_stds[max_class][1])
            if max_value > threshold:
                preds.append(max_class)
            else:
                preds.append(data.unseen_token_id)

        return np.array(preds)

    def get_prob_label(self, data, dataloader):

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask)
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs = torch.sigmoid(total_logits.detach())
        y_prob = total_probs.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        return y_true, y_prob
    
    def get_mu_stds(self, args, data):

        dataloader = data.train_dataloader
        y_true, y_prob = self.get_prob_label(data, dataloader)
        mu_stds = self.cal_mu_std(y_prob, y_true, data.num_labels)
    
        return mu_stds

    def evaluation(self, args, data,  mu_stds = None, show=False):
          
        y_true, y_prob = self.get_prob_label(data, data.test_dataloader)

        y_pred = self.classify_doc(data, args, y_prob, self.best_mu_stds)

        # self.predictions = list([data.label_list[idx] for idx in y_pred])
        # self.true_labels = list([data.label_list[idx] for idx in y_true])
        
        cm = confusion_matrix(y_true,y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc
        self.test_results = results

        if show:
            print(results)
        return (y_pred, y_true,)

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
                    loss_fct = nn.CrossEntropyLoss()
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode='train', loss_fct=loss_fct)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            mu_stds = self.get_mu_stds(args, data)
            
            y_true, y_prob = self.get_prob_label(data, data.eval_dataloader)
            y_pred = self.classify_doc(data, args, y_prob, mu_stds = mu_stds)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score 
                self.best_mu_stds = mu_stds
            else:
                print(wait)
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model 

        if args.save_detect:
            self.save_model(args)
    
    def fit(self, prob_pos_X):
        prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std

    def cal_mu_std(self, probs, trues, num_labels):

        mu_stds = []
        for i in range(num_labels):
            pos_mu, pos_std = self.fit(probs[trues == i, i])
            mu_stds.append([pos_mu, pos_std])

        return mu_stds

    def freeze_bert_parameters(self, model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def restore_model(self, args):
        output_model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    def save_model(self, args):

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(self.model_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())  
    
    
    def save_results(self, args, data = None):
        
        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        #save known intents
        np.save(os.path.join(self.output_file_dir, 'labels.npy'), data.label_list)
        np.save(os.path.join(self.output_file_dir, 'mu_stds.npy'),self.best_mu_stds)

        save_csv_json(args, data, self.true_labels, self.predictions)
        





  

    
    
