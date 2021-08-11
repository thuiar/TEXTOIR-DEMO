
import argparse
import logging
import os
import datetime
import sys
import json
import copy
import numpy as np

from pipeline.configs.base import ParamManager
from pipeline.utils.functions import save_pipeline_results, save_final_results, save_json_results, save_numpy_results, combine_test_results
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

    parser.add_argument("--config_file_name", type=str, default='ADB.py', help = "The name of the detection config file.")

    parser.add_argument("--train", action="store_true", help="Whether train the detection model.")

    parser.add_argument("--backbone", type=str, default='bert', help="which backbone to use")

    parser.add_argument("--save_model", action="store_true", help="Whether save the detection model.")

    parser.add_argument("--save_results", action="store_true", help="Whether save the detection results.")

    parser.add_argument("--save_frontend_results", action="store_true", help="save final frontend results for open intent recognition.")

    parser.add_argument('--setting', type=str, default='semi_supervised', help="Type for clustering methods.")

    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")

    parser.add_argument("--model_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--output_dir", default= '/home/sharing/disk2/zhanghanlei/save_data_162/TEXTOIR/outputs', type=str, 
                        help="The output directory where all train data will be written.")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--log_dir", type=str, default='logs', help = "The directory of logs.")

    parser.add_argument('--logger_name', type=str, default='pipeline', help="Logger name for open intent detection.")

    parser.add_argument("--exp_dir", type=str, default = 'exp', help="The path to save experimental results")

    parser.add_argument("--exp_name", type=str, default = 'ADB_DeepAligned', help="The experimental name.")

    parser.add_argument("--frontend_result_dir", type=str, default = sys.path[0] + '/frontend/static/jsons', help="The path to save results")

    parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'pipeline_results.csv', help="The file name of all the results.")

    args = parser.parse_args()

    return args

def set_logger(args):
    

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.method}_{args.dataset}_{args.known_cls_ratio}_{args.labeled_ratio}_{args.seed}_{time}.log"
    
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
    sys.path.insert(0, args.working_path)

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

        logger.info('Results saved in %s', str(args.result_dir))
        from open_intent_detection.utils.functions import save_results
        save_results(args, copy.copy(outputs))

        output_dir = os.path.join(args.exp_dir, args.type + '.npy')
        save_numpy_results(output_dir, outputs['y_pred'])


    logger.info('Save pipeline results begin...')
    save_pipeline_results(args, data, method, outputs)
    logger.info('Save pipeline results finished...')

    if args.save_frontend_results:

        logger.info('Save detection frontend results begin...')
        from open_intent_detection.utils.frontend_analysis import save_analysis_table_results
        save_analysis_table_results(args, data, outputs, pipeline = True, save_dir = 'open_intent_recognition')
        logger.info('Save detection frontend results begin...')

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
    data_properties = data.get_attrs()
    save_properties = {}
    property_list = ['num_labels', 'n_known_cls', 'known_label_list', 'all_label_list', 'unseen_token_id']

    for i, key in enumerate(property_list):
        save_properties[key] = data_properties[key]

    sys.path.insert(0, args.working_path)
    
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
        
        logger.info('Results saved in %s', str(args.result_dir))
        from open_intent_discovery.utils.functions import save_results
        save_results(args, copy.copy(outputs))

        output_json_dir = os.path.join(args.exp_dir, args.type + '_outputs.json')
        
        save_json_results(output_json_dir, outputs)
        output_data_dir = os.path.join(args.exp_dir, args.type + '_data.json')
        save_json_results(output_data_dir, save_properties)

    if args.save_frontend_results:
        
        logger.info('Save discovery frontend results begin...')
        from open_intent_discovery.utils.frontend_analysis import save_analysis_table_results
        save_analysis_table_results(args, data, outputs, args.logger_name, pipeline = True, save_dir = 'open_intent_recognition')
        logger.info('Save discovery frontend results finished...')

def run():
    
    args = parse_arguments()

    args.exp_name = '_'.join([str(x) for x in [args.exp_name, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]])
    args.exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    logger = set_logger(args)
    
    if args.type == 'Detection':
        run_detect(args, logger)
    
    elif args.type == 'Discovery':
        run_discover(args, logger)

    elif args.type == 'Pipeline':

        logger.info('This is the pipeline for open intent detection and discovery...')

        detection_load_path = os.path.join(args.exp_dir, 'Detection.npy')
        with open(detection_load_path, 'r')  as f:
            detection_preds = np.load(detection_load_path)

        outputs_load_path = os.path.join(args.exp_dir, 'Discovery_outputs.json')
        with open(outputs_load_path, 'r')  as f:
            discovery_results = json.load(f)  
        
        data_load_path = os.path.join(args.exp_dir, 'Discovery_data.json')
        with open(data_load_path, 'r')  as f:
            discovery_data = json.load(f)          
        
        final_results = combine_test_results(args, detection_preds,\
            discovery_data, discovery_results, logger)

        logger.info("***** Final results *****")
        for key in sorted(final_results.keys()):
            logger.info("  %s = %s", key, str(final_results[key]))
        
        if args.save_results:
            save_final_results(args, final_results)

if __name__ == '__main__':

    run()
