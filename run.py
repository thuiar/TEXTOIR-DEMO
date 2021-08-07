
import argparse
import logging
import os
import datetime
import sys
import copy

from keybert import KeyBERT
from pipeline.configs.base import ParamManager
from pipeline.utils.functions import save_pipeline_results, combine_test_results
from pipeline.dataloaders.base import Data_Detection, Data_Discovery

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.path.join(sys.path[0], 'data'), help="Data Directory.") 

    parser.add_argument('--type', type=str, default='pipeline', help="Type for methods")

    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
    
    parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
    
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--method", type=str, default='ADB', help="which detection method to use")

    parser.add_argument("--config_detection_file", type=str, default='ADB.py', help = "The name of the detection config file.")

    parser.add_argument("--train", action="store_true", help="Whether train the detection model.")

    parser.add_argument("--backbone", type=str, default='bert', help="which backbone to use")

    parser.add_argument("--save_model", action="store_true", help="Whether save the detection model.")

    parser.add_argument("--save_results", action="store_true", help="Whether save the detection results.")

    parser.add_argument("--model_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    # parser.add_argument("--discovery_method", type=str, default='DeepAligned', help="which discovery method to use")

    parser.add_argument("--config_discovery_file", type=str, default='DeepAligned.py', help = "The name of the discovery config file.")

    parser.add_argument("--output_dir", default= '/home/sharing/disk2/zhanghanlei/save_data_162/TEXTOIR/outputs', type=str, 
                        help="The output directory where all train data will be written.")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--log_dir", type=str, default='logs', help = "The directory of logs.")

    parser.add_argument('--log_id', type=str, default='1', help="Training record ID.")

    parser.add_argument('--logger_name', type=str, default='pipeline', help="Logger name for open intent detection.")

    parser.add_argument("--pipe_data_dir", type=str, default = 'pipe_data', help="The path to save results")

    parser.add_argument("--frontend_result_dir", type=str, default = '/frontend/static/jsons', help="The path to save results")

    parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'pipeline_results.csv', help="The file name of all the results.")

    args = parser.parse_args()

    return args

def set_logger(args):
    

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{time}.log"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def run_detect(args, logger):
    

    param = ParamManager(args, type = 'detection')
    args = param.args
    
    logger.info('Open Intent Detection Begin...')
    logger.info('Parameters Initialization...')
    
    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    logger.info('Data and Model Preparation...')
    
    data = Data_Detection(args)
    
    working_path = '/home/zhanghanlei/git/submit/TEXTOIR-DEMO/open_intent_detection'
    sys.path.insert(0, working_path)

    from open_intent_detection.backbones.base import ModelManager as detection_model
    model = detection_model(args, data, logger_name = 'Detection')
    
    from open_intent_detection.methods import method_map
    
    method_manager = method_map[args.method]
    method = method_manager(args, data, model, logger_name = args.logger_name)

    if args.train:
        
        logger.info('Training Begin...')
        method.train(args, data)
        logger.info('Training Finished...')
    
    logger.info('Testing begin...')
    outputs = method.test(args, data)
    logger.info('Testing finished...')

    if args.save_results:
        logger.info('Results saved in %s', str(os.path.join(args.result_dir, args.type)))
        from open_intent_detection.utils.functions import save_results
        save_results(args, copy.copy(outputs))

    logger.info('Save pipeline results begin...')
    save_pipeline_results(args, data, method, outputs)
    logger.info('Save pipeline results finished...')
    
    return outputs


def run_discover(args, logger):

    param = ParamManager(args, type = 'discovery')
    args = param.args

    logger.info('Open Intent Discovery Begin...')
    logger.info('Parameters Initialization...')

    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    logger.info('Data and Model Preparation...')

    data = Data_Discovery(args, logger_name = args.logger_name)
    
    working_path = '/home/zhanghanlei/git/submit/TEXTOIR-DEMO/open_intent_discovery/'
    sys.path.insert(0, working_path)
    
    from open_intent_discovery.backbones.base import ModelManager
    model = ModelManager(args, data)


    from open_intent_discovery.methods import method_map
    method_manager = method_map[args.method]
    method = method_manager(args, data, model, logger_name = args.logger_name)

    if args.train:
            
        logger.info('Training Begin...')
        method.train(args, data)
        logger.info('Training Finished...')

    logger.info('Testing begin...')
    outputs = method.test(args, data)
    logger.info('Testing finished...')

    if args.save_results:
        logger.info('Results saved in %s', str(os.path.join(args.results_dir, args.type)))
        save_results(args, copy.copy(outputs))
        
    return outputs, data


def run():
    
    args = parse_arguments()
    
    logger = set_logger(args)
    # logger.info('This is the pipeline for open intent detection and discovery...')
    
    if args.type == 'Detection':
        detection_results = run_detect(args, logger)
    
    elif args.type == 'Discovery':
        discovery_results, discovery_data = run_discover(args, logger)

    # else:
    #     final_results = combine_test_results(args, detection_results,\
    #     discovery_data, discovery_results, logger)
    
    # save_results(args, final_results)

if __name__ == '__main__':

    run()
