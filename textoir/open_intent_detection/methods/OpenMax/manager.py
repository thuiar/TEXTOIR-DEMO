from open_intent_detection.utils import *
from open_intent_detection.pretrain import *
from .openmax_utils import *
import open_intent_detection.Backbone as Backbone

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
        else:
            self.weibull_model = None
            self.restore_model(args)

    def cal_vec_dis(self, args, data, centroids, y_logit, y_true):
        mean_vectors = [x for x in centroids]

        dis_all = []
        for i in range(data.num_labels):
            arr = y_logit[y_true == i]
            dis_all.append(self.distance_compute(args, arr, mean_vectors[i]))
        dis_sorted = [sorted(x) for x in dis_all]

        return mean_vectors, dis_sorted

    def classify_openmax(self, data, args, num_samples, y_prob, y_logit):
        
        y_pred = []
        for i in range(num_samples):
            textarr = {}
            textarr['scores'] = y_prob[i]
            textarr['fc8'] = y_logit[i]
            openmax, softmax = recalibrate_scores(self.weibull_model, data.num_labels, textarr, alpharank=min(args.alpharank, data.num_labels))
            openmax = np.array(openmax)
            pr = np.argmax(openmax)
            p = max(openmax)

            if p < args.threshold:
                pr = data.unseen_token_id
            y_pred.append(pr)    

        return y_pred

    def get_pred_label(self, data, dataloader, compute_centroids=False):
        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        centroids = torch.zeros(data.num_labels, data.num_labels).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask)
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
            
                if compute_centroids:
                    for i in range(len(label_ids)):
                        centroids[label_ids[i]] += logits[i]  

        _, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        total_probs= F.softmax(total_logits.detach(), dim=1)
        
        y_true = total_labels.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_prob = total_probs.cpu().numpy()
        y_logit = total_logits.cpu().numpy()

        if compute_centroids:
            
            centroids /= torch.tensor(self.class_count(y_true)).float().unsqueeze(1).to(self.device)
            centroids = centroids.detach().cpu().numpy()
            return y_true, y_pred, y_prob, y_logit, centroids

        else:
            return y_true, y_pred, y_prob, y_logit

    def evaluation(self, args, data, dataloader, show = False):
            
        y_true, y_pred, _, y_logit, centroids = self.get_pred_label(data, data.train_dataloader, compute_centroids=True)

        mean_vecs, dis_sorted = self.cal_vec_dis(args, data, centroids, y_logit, y_true)
        self.weibull_model = weibull_tailfitting(mean_vecs, dis_sorted, data.num_labels, tailsize = args.weibull_tail_size)

        y_true, _, y_prob, y_logit = self.get_pred_label(data, dataloader)
        y_pred = self.classify_openmax(data, args, len(y_true), y_prob, y_logit)

        # self.predictions = list([data.label_list[idx] for idx in y_pred])
        # self.true_labels = list([data.label_list[idx] for idx in y_true])
        
        cm = confusion_matrix(y_true,y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc
        self.test_results = results

        if show:
            print(results)
        return y_pred, y_true, y_prob
        


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
            y_true, y_pred, _, _ = self.get_pred_label(data, data.eval_dataloader)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            print('eval_score',eval_score)
            
            
            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        self.model = best_model
        
        if args.save_detect: 
            self.save_model(args)
    
    def distance_compute(self, args, arr, mav):
        pre = []
        for i in arr:
            pre.append(compute_distance(i, mav,args.distance_type))
        return pre

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def freeze_bert_parameters(self, model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
        
    def restore_model(self, args):
        output_model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    
    def save_results(self, args, data = None):

        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        #save known intents
        np.save(os.path.join(self.output_file_dir, 'labels.npy'), data.label_list)

        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
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
        key = str(args.dataset) + '_' + str(args.known_cls_ratio) + '_' + str(args.labeled_ratio) + '_' + str(args.method)
        if os.path.exists(tf_overall_path):
            results = json_read(tf_overall_path)

        results[key] = predict_t_f

        if os.path.exists(tf_fine_path):
            results_fine = json_read(tf_fine_path)
        results_fine[key] = predict_t_f_fine

        json_add(results, tf_overall_path)
        json_add(results_fine, tf_fine_path)
        
        print('test_results', data_diagram)

    def save_model(self, args):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(self.model_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())  

        


    



  

    
    
