import argparse

class Param:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser = self.common_param(parser)
        args = parser.parse_args()  

        backbones = {'bert':self.bert, 'unsup':self.unsup}
        methods = {'DEC':self.DEC, 'DCN':self.DCN, 'DTC_BERT':self.DTC_BERT}
        parser = backbones[args.backbone](parser)
        
        if args.method in methods:
            parser = methods[args.method](parser, args)

        self.args = parser.parse_args()  

    def common_param(self, parser):

        parser.add_argument("--dataset", default=None, type=str, help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--method", type=str, default='DeepAligned', help="which method to use")

        parser.add_argument("--backbone", default='bert', type=str, help="which model to use")

        parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        
        parser.add_argument('--type', type=str, default='open_intent_discovery', help="Type for methods")

        parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
        
        parser.add_argument('--setting', type=str, default='unsupervised', help="Type for clustering methods.")

        parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")

        parser.add_argument("--data_dir", default='data', type=str,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
         
        parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

        return parser

    def unsup(self, parser):

        parser.add_argument("--max_num_words", default=10000, type=int, help="The maximum number of words.")
        
        parser.add_argument("--feat_dim", default=2000, type=int, help="The feature dimension.")

        parser.add_argument("--glove_model", default="/home/zhl/pretrained_models/glove", type=str, help="The path for the pre-trained bert model.")

        return parser


    def DEC(self, parser, args = None):

        parser.add_argument("--maxiter", default=1, type=int, help="The training epochs for DEC.")

        parser.add_argument("--update_interval", default=100, type=int, help="The training epochs for DEC.")

        parser.add_argument("--batch_size", default=256, type=int, help="The training epochs for DEC.")

        parser.add_argument("--tol", default=0.001, type=float, help="The tolerance threshold to stop training for DEC.")

        return parser
    
    def DCN(self, parser, args = None):

        parser.add_argument("--maxiter", default=1, type=int, help="The training epochs for DCN.")

        parser.add_argument("--update_interval", default=100, type=int, help="The training epochs for DCN.")

        parser.add_argument("--batch_size", default=256, type=int, help="The training epochs for DCN.")

        parser.add_argument("--tol", default=0.001, type=float, help="The tolerance threshold to stop training for DCN.")
    
        return parser

    # def DTC_BERT(self, parser, args = None):

    #     parser.add_argument("--rampup_coefficient", default=10.0, type=float, help="The rampup coefficient.")

    #     parser.add_argument("--rampup_length", default=5, type=int, help="The rampup length.")

    #     parser.add_argument("--num_warmup_train_epochs", default=10, type=int, help="The number of warm-up training epochs.")

    #     parser.add_argument("--alpha", default=0.6, type=float)

    #     args.lr_pre = 2e-5

    #     return parser

    # def CDAC+(self, parser, args = None):

    #     args.num_train_epochs = 46
    #     args.train_batch_size = 256
    #     args.eval_batch_size = 256        
    #     args.learning_rate = 5e-5


    def bert(self, parser):
        ##############Your Location for Pretrained Bert Model#####################
        parser.add_argument("--bert_model", default="/home/zhl/pretrained_models/uncased_L-12_H-768_A-12", type=str, help="The path for the pre-trained bert model.")

        parser.add_argument("--pretrain", action="store_true", default = 'pretrain', help="Pretrain the model")

        parser.add_argument("--pretrain_dir", default='pretrain_models', type=str, 
                            help="The output directory where the model checkpoints will be written.") 
        
        parser.add_argument("--max_seq_length", default=None, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")

        parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

        parser.add_argument("--warmup_proportion", default=0.1, type=float)

        parser.add_argument("--freeze_bert_parameters", action="store_true", default = True, help="Freeze the last parameters of BERT")

        parser.add_argument("--save_results", action="store_true", default = 'save_results', help="save test results")
        
        parser.add_argument("--save_model", action="store_true", help="save trained-model")

        parser.add_argument("--lr", default=5e-5, type=float,
                            help="The learning rate of BERT.")    

        parser.add_argument("--lr_pre", default=5e-5, type=float, help="The learning rate for pre-training.")

        parser.add_argument("--num_train_epochs", default=100.0, type=float,
                            help="Total number of training epochs to perform.") 
        
        parser.add_argument("--train_batch_size", default=128, type=int,
                            help="Batch size for training.")
        
        parser.add_argument("--eval_batch_size", default=64, type=int,
                            help="Batch size for evaluation.")    
        
        parser.add_argument("--wait_patient", default=20, type=int,
                            help="Patient steps for Early Stop.")    
        
        return parser

class DTC_BERT(Param):

        parser = self.parser

        parser.add_argument("--rampup_coefficient", default=10.0, type=float, help="The rampup coefficient.")
        
        parser.add_argument("--rampup_length", default=5, type=int, help="The rampup length.")

        parser.add_argument("--num_warmup_train_epochs", default=10, type=int, help="The number of warm-up training epochs.")

        parser.add_argument("--alpha", default=0.6, type=float)

