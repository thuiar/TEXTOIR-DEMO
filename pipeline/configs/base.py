import argparse
import sys
import os
import importlib
from easydict import EasyDict

class ParamManager:
    
    def __init__(self, args, type = None): 

        if type == 'detection':

            config_file_name = args.config_detection_file
            method_common_param, method_hyper_param = self.get_method_param(config_file_name, type)
            method_common_param['method'] = args.detection_method
            method_common_param['train'] = args.detection_train
            method_common_param['save_model'] = args.detection_save_model
            method_common_param['save_results'] = args.detection_save_results

        elif type == 'discovery':

            config_file_name = args.config_discovery_file
            method_common_param, method_hyper_param = self.get_method_param(config_file_name, type)
            method_common_param['method'] = args.discovery_method
            method_common_param['train'] = args.discovery_train
            method_common_param['save_model'] = args.discovery_save_model
            method_common_param['save_results'] = args.discovery_save_results
        
        else:
            print('This type of OIR is not implemented, please check if the spell is correct.')

        method_common_param['known_cls_ratio'] = args.known_cls_ratio
        method_common_param['dataset'] = args.dataset
        method_common_param['labeled_ratio'] = args.labeled_ratio

        args = EasyDict(
                            dict(
                                vars(args),
                                **method_common_param, 
                            )
                        )

        output_path_param = self.add_output_path_param(args)

        self.args = EasyDict(
                                dict(
                                        **output_path_param,
                                        **vars(args), 
                                        **method_hyper_param
                                    )
                            )

    def get_method_param(self, config_file_name, type):
        
        if config_file_name.endswith('.py'):
            module_name = "pipeline.configs." + str(type) + '.' + str(config_file_name[:-3])
        else:
            module_name = "pipeline.configs." + str(type) + '.' +  str(config_file_name)

        config = importlib.import_module(module_name)

        method_param = config.Param
        method_args = method_param()

        return method_args.common_param, method_args.hyper_param

    def add_output_path_param(self, args):
        
        task_output_dir = os.path.join(args.output_dir, args.type)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone, args.seed]
        method_output_name = "_".join([str(x) for x in concat_names])

        method_output_dir = os.path.join(task_output_dir, method_output_name)
        if not os.path.exists(method_output_dir):
            os.makedirs(method_output_dir)

        model_output_dir = os.path.join(method_output_dir, args.model_dir)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        output_path_param = {
            'method_output_dir': method_output_dir,
            'model_output_dir': model_output_dir,
        }

        return output_path_param