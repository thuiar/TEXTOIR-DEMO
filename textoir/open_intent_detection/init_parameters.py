import argparse
import sys
import os

class Param:

    def __init__(self, args_pipe=None):

        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        args_dict = all_args.__dict__

        if args_pipe is not None:
            args_pipe_dict = args_pipe.__dict__
            for key in args_pipe_dict:
                if key in args_dict.keys():
                    args_dict[key] = args_pipe_dict[key]
                else:
                    args_dict[key] = args_pipe_dict[key]

        # method = eval(all_args.method)
        try:
            mtd = all_args.method == all_args.detect_method
        except:
            mtd = True
        if mtd:
            method = eval(all_args.method)
            print("only detect or same with pipline")
        else:
            method = eval(all_args.detect_method)
            print("pipline detect")
        self.args = method(all_args).args
        
    
    def all_param(self, parser):

        ##################################common parameters####################################
        parser.add_argument("--dataset", default='banking', type=str, 
                            help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--method", type=str, default='ADB', help="which method to use")

        parser.add_argument("--backbone", type=str, default='bert', help="which model to use")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument('--type', type=str, default='open_intent_detection', help="Type for methods")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")

        parser.add_argument("--num_train_epochs", default=100.0, type=float,
                            help="Total number of training epochs to perform.") 

        parser.add_argument("--train_detect", action="store_true", help="Whether train the model")

        parser.add_argument("--frontend_dir", type=str, default=os.path.join(sys.path[0],'../frontend/static/jsons') , help="the path of the frontend")
        ###########################special parameters######################################
        #ADB
        parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")
        
        parser.add_argument("--threshold", type=float, default=0.5, help="The probability threshold.")
        #DOC
        parser.add_argument("--scale", type=float, default=2, help="The scale factor of DOC.")
        #DeepUnk
        parser.add_argument("--n_neighbors", type=int, default=20, help="The number of neighbors of LOF.")

        parser.add_argument("--contamination", type=float, default=0.05, help="The contamination factor of LOF.")
        #OpenMax
        parser.add_argument("--weibull_tail_size", type=int, default=20, help="The factor of weibull model.")

        parser.add_argument("--alpharank", type=int, default=10, help="The factor of alpha rank.")

        parser.add_argument("--distance_type", type=str, default='cosine', help="The distance type.")
        
        #######################################################bert############################################################################
        parser.add_argument("--bert_model", default="/home/sharing/disk2/zhl_backup/pretrained_models/uncased_L-12_H-768_A-12", type=str, help="The path for the pre-trained bert model.")
        
        parser.add_argument("--data_dir", default=sys.path[0]+'/data', type=str,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
        
        parser.add_argument("--output_path", type=str, default='outputs', help="the path to save output data")

        parser.add_argument("--model_dir", default='models', type=str, 
                            help="The output directory where the model predictions and checkpoints will be written.") 

        parser.add_argument("--train_data_dir", default= os.path.join(sys.path[0], 'models'), type=str, 
                            help="The output directory where all train data will be written.") 

        parser.add_argument("--results_path", type=str, default = os.path.join(sys.path[0], 'results'), help="the path to save results")
        
        parser.add_argument("--max_seq_length", default=None, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")

        parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

        parser.add_argument("--warmup_proportion", default=0.1, type=float)

        parser.add_argument("--freeze_bert_parameters", action="store_true", default="freeze", help="Freeze the last parameters of BERT")

        parser.add_argument("--save_detect", action="store_true", help="save trained-model for open intent detection")
        
        parser.add_argument("--gpu_id", type=str, default='1', help="Select the GPU id")

        parser.add_argument("--lr", default=2e-5, type=float,
                            help="The learning rate of BERT.")    
        
        parser.add_argument("--train_batch_size", default=64, type=int,
                            help="Batch size for training.")
        
        parser.add_argument("--eval_batch_size", default=64, type=int,
                            help="Batch size for evaluation.")    
        
        parser.add_argument("--wait_patient", default=10, type=int,
                            help="Patient steps for Early Stop.") 

        return parser
        
class ADB:
    def __init__(self, args):
        self.args = args

class MSP:
    def __init__(self, args):
        self.args = args

class DOC:
    def __init__(self, args):
        self.args = args

class DeepUnk:
    def __init__(self, args):
        args.num_train_epochs = 200
        self.args = args

class OpenMax:
    def __init__(self, args):
        self.args = args
