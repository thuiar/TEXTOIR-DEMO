from open_intent_detection.init_parameters import Param
from open_intent_detection.dataloader import *
from open_intent_detection.utils import *

def run(args):

    print('Data and Model Preparation...')
    data = Data(args)        
    manager = get_manager(args, data)   
    
    if args.train_detect:
        
        print('Training Begin...')
        manager.train(args, data)
        print('Training Finished...')

    print('Evaluation begin...')
    outputs = manager.evaluation(args, data, data.test_dataloader)
    print('Evaluation finished...')

    if args.save_detect:
        save_detect_backend_results(manager, args, data)
    
    debug(outputs, data, manager, args)
    print('Open Intent Detection Finished...')

if __name__ == '__main__':
    print('Open Intent Detection Begin...')
    print('Parameters Initialization...')
    param = Param()
    args = param.args 
    
    run(args)
    

