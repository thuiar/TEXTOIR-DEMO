from open_intent_detection.init_parameters import Param
from open_intent_detection.dataloader import *
from open_intent_detection.utils import debug
import importlib

def run(args):
    # test = True
    # if test:
    #     print("Notice: ****This is test mode****")
    #     args.dataset = 'banking'
    #     args.labeled_ratio = 1
    #     args.known_cls_ratio = 0.25
    #     args.num_train_epochs = 1

    print('Data and Model Preparation...')
    data = Data(args)
    module_names = [args.type, 'methods', args.method, 'manager']
    import_name = ".".join(module_names)
    method = importlib.import_module(import_name) 
    manager = method.ModelManager(args, data)   

    if not os.path.exists(os.path.join(args.train_data_dir, args.type)):
        os.makedirs(os.path.join(args.train_data_dir, args.type))
    
    if args.train:
        
        print('Training Begin...')
        manager.train(args, data)
        print('Training Finished...')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished...')

    manager.save_results(args, data)
    
    debug(data, manager, args)
    print('Open Intent Detection Finished...')
    
if __name__ == '__main__':
    
    print('Open Intent Detection Begin...')

    print('Parameters Initialization...')
    # param = Param()

    # args = param.args 
    
    # run(args)

    datasets = ['oos', 'stackoverflow', 'banking']
    known_cls_ratios = [0.25, 0.5, 0.75]
    labeled_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = [i for i in range(10)]
    

    for seed in seeds:
        for dataset in datasets:
            for known_cls_ratio in known_cls_ratios:
                for labeled_ratio in labeled_ratios:
                    param = Param()
                    args = param.args 
        
                    args.method = 'DeepUnk'
                    args.dataset = dataset
                    args.known_cls_ratio = known_cls_ratio
                    args.labeled_ratio = labeled_ratio
                    args.seed = seed 
                    args.gpu_id = '1'
                    args.train = True
                    args.freeze_bert_parameters = True

                    run(args)