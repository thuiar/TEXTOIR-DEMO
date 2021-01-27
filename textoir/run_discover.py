from open_intent_discovery.init_parameters import Param
from open_intent_discovery.dataloader import *
from open_intent_discovery.utils import debug
import importlib

def run(args, inputs=None):
    test = True
    if test:
        print("Notice: ****This is test mode****")
        args.dataset = 'banking'
        args.labeled_ratio = 0.1
        args.known_cls_ratio = 0.25
        args.num_train_epochs = 1

    print('Data Preparation...')
    data = Data(args)
    method = importlib.import_module('open_intent_discovery.methods.' + args.method + '.manager')
    manager = method.ModelManager(args, data)   
    

    print('Training Begin...')
    manager.train(args, data)
    print('Training Finished...')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished...')

    debug(data, manager, args)
    print('Open Intent Discovery Finished...')

if __name__ == '__main__':
    print('Open Intent Discovery Begin...')

    print('Parameters Initialization...')
    param = Param()
    args = param.args 

    run(args)