from numpy.core.numeric import False_
from open_intent_detection.init_parameters import Param as Param_detect
from pipeline.dataloader import Data_detect
from open_intent_detection.utils import *

from open_intent_discovery.init_parameters import Param as Param_discover
from open_intent_discovery.utils import *

from pipeline.init_parameters import Param as Param_pipe
from pipeline.manager import PipelineManager
from pipeline.dataloader import Data_discover, Unsup_Data_Discover
from pipeline.utils import *
import operator
from keybert import KeyBERT

def run_detect(args, data):

    manager = get_manager(args, data, 'detection')   

    if args.train_detect: 
        
        print('Open Intent Detection: Training Begin...')
        manager.train(args, data)
        print('Open Intent Detection: Training finished...')
        
    print('Open Intent Detection: Prediction begin...')
    outputs = predict_save_detection_results(args, data, manager)
    print('Open Intent Detection: Prediction finished...')

    return outputs

def run_discover(args, data, inputs=None):

    manager = get_manager(args, data, 'discovery')   
    
    if args.train_discover:
        print('Training Begin...')
        manager.train(args, data)
        print('Training Finished...')

    outputs = manager.evaluation(args, data)
    test_label_ids, test_true_ids, feats = outputs[0], outputs[1], outputs[2]
    
    ids_known = [idx for idx, label in enumerate(inputs) if label != data.unseen_token_id]
    ids_open = [idx for idx, label in enumerate(inputs) if label == data.unseen_token_id]
    open_feats = feats[ids_open]
    k = data.num_labels - data.n_known_cls
    if args.setting == "semi_supervised":
        all_label_list = data.all_label_list
        km = KMeans(n_clusters=k, n_jobs=-1, random_state = args.seed)
        km.fit(open_feats)
        open_labels = km.labels_ + data.n_known_cls
    elif args.setting == "unsupervised":
        all_label_list = data.le.classes_
        #### unsupervised model
        #### use k-means models 
        use_k_means_models = ['DEC', 'DCN', 'KM', 'SAE']
        use_ag = ['AG']
        if args.method in use_k_means_models:
            km = KMeans(n_clusters=k, n_jobs=-1, random_state = args.seed)
            km.fit(open_feats)
            open_labels = km.labels_ + data.n_known_cls
        elif args.method in use_ag:
            ag = manager.ag
            fit_pred = ag.fit_predict(open_feats)
            open_labels = fit_pred + data.n_known_cls
    if isinstance(inputs, list):
        inputs = np.array(inputs)
    test_label_ids[ids_known] = inputs[ids_known]
    test_label_ids[ids_open] = open_labels 
    
    # keyword_extraction(args, data, test_label_ids, ids_open)

    test_known_ids = [idx for idx, label in enumerate(test_true_ids) if all_label_list[label] in data.known_label_list]

    kn_acc = accuracy_score(test_true_ids[test_known_ids], test_label_ids[test_known_ids])
    kn_f1 = f1_score(test_true_ids[test_known_ids], test_label_ids[test_known_ids], average="macro")
    print('known acc', kn_acc)
    print('known_f1', kn_f1)

    test_open_ids = [idx for idx, label in enumerate(test_true_ids) if all_label_list[label] not in data.known_label_list]
    open_results = clustering_score(test_true_ids[test_open_ids], test_label_ids[test_open_ids])
    print('open_results', open_results)

    all_results = {}
    all_results['known_acc'] = kn_acc
    all_results['known_f1'] = kn_f1
    all_results['open_nmi'] = open_results['NMI']
    all_results['open_ari'] = open_results['ARI']
    
    if args.save_discover:
        save_pipeline_backend_results(all_results, args, data)

if __name__ == '__main__':
    # detect_method = ['MSP', 'DOC', 'DeepUnk', 'ADB']
    detect_method = ['DOC']
    # detect_method = ['ADB']
    discover_method = ['DeepAligned']
    # datasets = ['clinc']
    datasets = ['stackoverflow']
    # datasets = ['clinc', 'snips', 'banking', 'stackoverflow']
    settings = ["semi_supervised", "unsupervised"]
    for dt in datasets:
        for det in detect_method:
            train_det = False
            # if dt in ['clinc'] and det in ['MSP', 'DOC']:
            #     continue
            # if dt in ['banking'] and det in ['MSP']:
            #     continue
            for dis in discover_method:
                if dis in ['CDACPlus', 'DeepAligned']:
                    setting_type = "semi_supervised"
                else:
                    setting_type = "unsupervised"
                
                torch.cuda.set_device(1)
                
                param_pipe = Param_pipe()
                args_pipe = param_pipe.args
                args_pipe.pipe_results_path = os.path.join(args_pipe.type, args_pipe.pipe_results_path)

                print("Notice: ****Train settings for pipeline****")
                args_pipe.dataset = dt
                args_pipe.labeled_ratio = 0.75
                args_pipe.known_cls_ratio = 0.5
                args_pipe.num_train_epochs = 100
                args_pipe.detect_method = det
                args_pipe.discover_method = dis
                args_pipe.setting = setting_type
                args_pipe.train = False
                args_pipe.train_detect = False
                args_pipe.train_discover = True

                print('Open Intent Detection: Data and Parameters Preparation...')
                param_detect = Param_detect(args_pipe)
                args_detect = param_detect.args
                args_detect.method = args_pipe.detect_method
                data_detect = Data_detect(args_detect)
                outputs = run_detect(args_detect, data_detect)

                print('Open Intent Discovery: Data and Parameters Preparation...')
                param_discover = Param_discover(args_pipe)
                args_discover = param_discover.args
                args_discover.method = args_pipe.discover_method
                if args_discover.setting == "semi_supervised":
                    data_discover = Data_discover(args_discover)
                else:
                    data_discover = Unsup_Data_Discover(args_discover)
                run_discover(args_discover, data_discover, outputs)

                torch.cuda.empty_cache()
                train_det = False
