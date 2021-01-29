import argparse

class Param:

    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser = self.common_param(parser)
        self.args = parser.parse_args()  

    def common_param(self, parser):

        parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--detect_method", type=str, default='ADB', help="which method to use")

        parser.add_argument("--discover_method", type=str, default='DeepAligned', help="which method to use")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
        
        parser.add_argument('--type', type=str, default='pipeline', help="Type for methods")

        return parser