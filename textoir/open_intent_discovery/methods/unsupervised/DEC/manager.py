from open_intent_discovery.utils import *
from open_intent_discovery.pretrain import *
import open_intent_discovery.Backbone as Backbone

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

class ModelManager:
    
    def __init__(self, args, data):
        
        self.tfidf_train, self.tfidf_test = data.get_tfidf(args)
        self.sae_emb = Backbone.get_stacked_autoencoder(self.tfidf_train.shape[1])

        self.km = None
        self.y_test = None
        self.test_results = None
        self.num_labels = data.num_labels
        self.model = None

    def train(self, args, data):
        
        emb_train, emb_test = self.init_emb(args, data)
        
        clustering_layer = Backbone.ClusteringLayer(self.num_labels, name='clustering')(self.sae_emb.layers[3].output)
        model = Model(inputs=self.sae_emb.input, outputs=clustering_layer)
        model.compile(optimizer=SGD(0.001, 0.9), loss='kld')
        km = KMeans(n_clusters=self.num_labels, n_init=20, n_jobs=-1)
        
        y_pred = km.fit_predict(emb_train)
        y_pred_last = np.copy(y_pred)
        model.get_layer(name='clustering').set_weights([km.cluster_centers_])

        index = 0
        loss = 0
        index_array = np.arange(self.tfidf_train.shape[0])

        for ite in range(int(args.maxiter)):

            if ite % args.update_interval == 0:
                q = model.predict(self.tfidf_train, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if data.y_train is not None:
                    results = clustering_score(data.y_train, y_pred)
                    print('Iter=', ite, results, 'loss=', np.round(loss, 5))
                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < args.tol:
                    print('delta_label ', delta_label, '< tol ', args.tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * args.batch_size: min((index+1) * args.batch_size, self.tfidf_train.shape[0])]
            loss = model.train_on_batch(x=self.tfidf_train[idx], y=p[idx])
            index = index + 1 if (index + 1) * args.batch_size <= self.tfidf_train.shape[0] else 0

        self.model = model

    def init_emb(self, args, data):
        
        emb_train, emb_test = data.get_sae(args, self.sae_emb, self.tfidf_train, self.tfidf_test)
        km = KMeans(n_clusters= self.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(emb_train)

        return emb_train, emb_test


    def evaluation(self, data, show=False):
        q = self.model.predict(self.tfidf_test, verbose = 0)
        y_pred = q.argmax(1)
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
        
    
