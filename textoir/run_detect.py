from open_intent_detection.init_parameters import Param
from open_intent_detection.dataloader import *
from open_intent_detection.utils import debug
import importlib

def run(args):
    test = True
    if test:
        print("Notice: ****This is test mode****")
        args.dataset = 'banking'
        args.labeled_ratio = 1
        args.known_cls_ratio = 0.25
        args.num_train_epochs = 1

    print('Data Preparation...')
    data = Data(args)
    method = importlib.import_module('open_intent_detection.methods.' + args.method + '.manager')
    manager = method.ModelManager(args, data)   
    

    print('Training Begin...')
    manager.train(args, data)
    print('Training Finished...')

    print('Evaluation begin...')
    manager.evaluation(args, data, mode='test')
    print('Evaluation finished...')

    manager.save_results(args)
    
    debug(data, manager, args)
    print('Open Intent Detection Finished...')
    
if __name__ == '__main__':
    print('Open Intent Detection Begin...')

    print('Parameters Initialization...')
    param = Param()
    args = param.args 
    
    run(args)

    # datasets = ['oos', 'stackoverflow', 'banking']
    # known_cls_ratios = [0.25, 0.5, 0.75]
    # seeds = [i for i in range(10)]
    

    # for seed in seeds:
    #     for dataset in datasets:
    #         for known_cls_ratio in known_cls_ratios:
                
    #             args.num_train_epochs = 200
    #             args.dataset = dataset
    #             args.known_cls_ratio = known_cls_ratio
    #             args.labeled_ratio = 1.0
    #             args.seed = seed 
    #             args.gpu_id = '1'

    #             run(args)