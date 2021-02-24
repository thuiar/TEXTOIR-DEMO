from open_intent_discovery.utils import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):
    
        self.km = None
        self.emb_test = None
        self.y_test = None
        self.test_results = None
        self.num_labels = data.num_labels

    def train(self, args, data):

        emb_train, emb_test = data.get_glove(args, data.X_train, data.X_test)

        km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state = args.seed)
        km.fit(emb_train)

        self.km = km
        self.emb_test = emb_test

    def evaluation(self, data, show=False):
        y_pred = self.km.predict(self.emb_test)
        y_true = data.y_test
        results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true,y_pred) 

        if show:
            print('results',results)
            print('confusion matrix', cm)

        self.test_results = results

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
        
    
