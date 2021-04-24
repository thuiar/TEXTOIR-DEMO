import argparse

from numpy.core.numeric import False_

class Param:

    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser = self.common_param(parser)
        self.parser = parser 
        self.args = parser.parse_args()  

    def common_param(self, parser):

        parser.add_argument("--dataset", default='snips', type=str, help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--detect_method", type=str, default='DOC', help="which method to use")

        parser.add_argument("--discover_method", type=str, default='DEC', help="which method to use")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")

        parser.add_argument("--pipe_results_path_to_frontend", type=str, default='frontend/static/jsons/data_annotation', help="the path to save results of pipeline methods to  frontend")
        
        parser.add_argument('--type', type=str, default='pipeline', help="Type for methods")

        parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

        parser.add_argument("--train_detect", default=True, help="Whether train the model for open intent detection")

        parser.add_argument("--train_discover", default=True, help="Whether train the model for open intent discovery")

        parser.add_argument("--save_discover", default=True, help="save trained-model for open intent discovery")

        parser.add_argument("--save_detect", default=True, help="save trained-model for open intent detection")
        
        parser.add_argument('--setting', type=str, default='unsupervised', help="Type for clustering methods.")
        
        #semi_supervised
        return parser