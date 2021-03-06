from open_intent_detection.init_parameters import Param as Param_detect
from open_intent_discovery.init_parameters import Param as Param_discover
from pipeline.init_parameters import Param as Param_pipe
from open_intent_detection.dataloader import Data as Data_detect
from pipeline.dataloder import Data as Data_pipe
from run_detect import run as detect 
from run_discover import run as discover
from pipeline.manager import PipelineManager
import importlib
import os

def run_detect(args):

    print('Open Intent Detection Begin...')
    print('Parameters Initialization...')
    param_detect = Param_detect()
    args_detect = param_detect.args
    
    args_detect.dataset = args.dataset
    args_detect.known_cls_ratio = args.known_cls_ratio
    args_detect.labeled_ratio = args.labeled_ratio
    args_detect.seed = args.seed
    args_detect.method = args.detect_method
    args_detect.pipe_results_path = args.pipe_results_path
    args_detect.gpu_id = args.gpu_id

    test = True
    if test:
        print("Notice: ****This is test mode for open intent detection****")
        args_detect.dataset = 'banking'
        args_detect.labeled_ratio = 0.1
        args_detect.known_cls_ratio = 0.25
        args_detect.num_train_epochs = 1
    
    data = Data_pipe(args_detect, mode="detect")

    method_detect = importlib.import_module('open_intent_detection.methods.' + args_detect.method + '.manager')
    manager_detect = method_detect.ModelManager(args_detect, data)   
    print('Training Begin...')
    manager_detect.train(args_detect, data)
    print('Training Finished...')
    manager = PipelineManager()
    manager.detect(args_detect, data, manager_detect)

    manager.evaluation(args, data, mode='detect')

    return manager.predictions

def run_discover(args, inputs=None):
    print('Open Intent Discovery Begin...')
    print('Parameters Initialization...')
    param_discover = Param_discover()
    args_discover = param_discover.args 
    
    args_discover.dataset = args.dataset
    args_discover.known_cls_ratio = args.known_cls_ratio
    args_discover.labeled_ratio = args.labeled_ratio
    args_discover.seed = args.seed
    args_discover.method = args.discover_method
    args_discover.pipe_results_path = args.pipe_results_path
    args_discover.gpu_id = args.gpu_id

    test = True
    if test:
        print("Notice: ****This is test mode for open intent discovery****")
        args_discover.dataset = 'banking'
        args_discover.labeled_ratio = 0.1
        args_discover.known_cls_ratio = 0.25
        args_discover.num_train_epochs = 100
    
    # args_discover.data_dir = args.pipe_results_path
    data = Data_pipe(args_discover, mode='discover')

    method_discover = importlib.import_module('open_intent_discovery.methods.' + args_discover.method + '.manager')
    manager_discover = method_discover.ModelManager(args_discover, data)   

    print('Training Begin...')
    manager_discover.train(args_discover, data)
    print('Training Finished...')
    
    manager = PipelineManager(predictions=inputs)
    manager.discover(args_discover, data, manager_discover)

    manager.evaluation(args, data, mode='discover')

if __name__ == '__main__':

    param_pipe = Param_pipe()
    args_pipe = param_pipe.args
    args_pipe.pipe_results_path = os.path.join(args_pipe.type, args_pipe.pipe_results_path)

    outputs = run_detect(args_pipe)
    run_discover(args_pipe, outputs)
    