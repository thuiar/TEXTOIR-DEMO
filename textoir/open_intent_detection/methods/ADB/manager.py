from open_intent_detection.utils import *
from open_intent_detection.pretrain import *
import open_intent_detection.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels):
        
        logits = euclidean_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1) 
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()
        
        return loss, delta 
        
class ModelManager:
    
    def __init__(self, args, data):
        
        Model = Backbone.__dict__[args.backbone]
        self.model = Model.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

        self.test_results, self.predictions,  self.true_labels = None, None, None
        
        #Save models and trainined data
        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone]
        output_file_name = "_".join([str(x) for x in concat_names])
        output_dir = os.path.join(args.train_data_dir, args.type, output_file_name)
        self.output_file_dir = os.path.join(output_dir, args.type, args.save_results_path)
        self.model_dir = os.path.join(output_dir, args.model_dir)

        if args.train:
            
            self.best_eval_score = 0 
            self.delta = None
            self.delta_points = []
            self.centroids = None

        else:

            self.restore_model(args)
            self.delta = np.load(os.path.join(self.output_file_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).cuda()
            self.centroids = np.load(os.path.join(self.output_file_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).cuda()
        
    
    def pre_train(self, args, data):
        
        manager_p = PretrainModelManager(args, data)
        manager_p.model_dir = self.model_dir
        manager_p.train(args, data)
        print('Pretraining finished...')
        return manager_p.model

    def open_classify(self, data, features, delta, centroids):

        logits = euclidean_metric(features, centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= delta[preds]] = data.unseen_token_id

        return preds

    def get_pred_label(self, data, dataloader, delta, centroids):

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(data, pooled_output, delta, centroids)

                total_labels = torch.cat((total_labels,label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        return y_true, y_pred

    def evaluation(self, args, data):
            
        y_true, y_pred = self.get_pred_label(data, data.test_dataloader, self.delta, self.centroids)

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])
        
        cm = confusion_matrix(y_true,y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Acc'] = acc
        self.test_results = results


    def train(self, args, data):  

        self.model = self.pre_train(args, data)   
        
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim)
        delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        centroids = self.centroids_cal(args, data)

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
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, delta = criterion_boundary(features, centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(delta)

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            y_true, y_pred = self.get_pred_label(data, data.eval_dataloader, delta, centroids)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
            print('eval_score', eval_score)
            
            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                self.delta = copy.copy(delta)
                self.centroids = copy.copy(centroids)
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        
        return centroids

    def restore_model(self, args):
        output_model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    def cal_true_false(self):
        
        results = {}
        trues = np.array(self.true_labels)
        preds = np.array(self.predictions)

        for label in np.unique(trues):
            pos = np.array(np.where(trues == label)[0])
            num_pos = np.sum(preds[pos] == trues[pos])
            num_neg = np.sum(preds[pos] != trues[pos])

            results[label] = (str(num_pos), str(num_neg))
        
        return results

    def save_results(self, args, data):
        
        if not os.path.exists(self.output_file_dir):
            os.makedirs(self.output_file_dir)

        #save known intents
        np.save(os.path.join(self.output_file_dir, 'labels.npy'), data.label_list)

        #save true_false predictions
        predict_t_f = self.cal_true_false()

        with open(os.path.join(self.output_file_dir, 'ture_false.json'), 'w') as f:
            json.dump(predict_t_f, f)

        #save centroids, delta_points
        np.save(os.path.join(self.output_file_dir, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(self.output_file_dir, 'deltas.npy'), self.delta.detach().cpu().numpy())

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        result_file = 'results.csv'
        results_path = os.path.join(self.output_file_dir, result_file)
        
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

     




  

    
    
