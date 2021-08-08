from open_intent_detection.dataloaders.base import DataManager as Detection
from open_intent_discovery.dataloaders.base import DataManager as Discovery
from open_intent_discovery.dataloaders.bert_loader import get_loader
import numpy as np
import os
import copy
import logging

class Data_Detection(Detection):

    def __init__(self, args):
        super(Data_Detection, self).__init__(args)
        tmp_all_label_list = self.all_label_list
        self.all_label_list = copy.copy(self.known_label_list)

        for label in tmp_all_label_list:
            if label not in self.known_label_list:
                self.all_label_list.append(label)
        
        self.dataloader.test_true_examples = self.get_test_true_examples(args)

        if args.backbone == 'bert':
            from open_intent_detection.dataloaders.bert_loader import get_loader
            self.test_true_loader = get_loader(self.dataloader.test_true_examples, args, self.all_label_list, 'test')

    def get_test_true_examples(self, args):
        
        if args.backbone == 'bert':
            from open_intent_detection.dataloaders.bert_loader import DatasetProcessor
            processor = DatasetProcessor()
            ori_examples = processor.get_examples(self.data_dir, 'test')

        return ori_examples


class Data_Discovery(Discovery):

    def __init__(self, args, logger_name = 'Discovery'):

        super(Data_Discovery, self).__init__(args)

        logger = logging.getLogger(logger_name)

        self.data_dir = args.exp_dir
        self.known_label_list = list(np.load(os.path.join(args.exp_dir, 'known_labels.npy'), allow_pickle=True))
        self.all_label_list = list(np.load(os.path.join(args.exp_dir, 'all_labels.npy'), allow_pickle=True))

        self.n_known_cls = len(self.known_label_list)
        self.num_labels = int((len(self.all_label_list)) * args.cluster_num_factor)   
            
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        
        self.all_label_list.append(self.unseen_token)
        self.unseen_token_id = len(self.known_label_list)
        
        self.dataloader.train_labeled_examples, self.dataloader.train_unlabeled_examples \
            = self.get_pipe_examples(args, mode = 'train')
        logger.info("Number of labeled samples = %s", str(len(self.dataloader.train_labeled_examples)))
        logger.info("Number of unlabeled samples = %s", str(len(self.dataloader.train_unlabeled_examples)))

        self.dataloader.eval_examples = self.get_pipe_examples(args, mode = 'eval')
        self.dataloader.test_examples = self.get_pipe_examples(args, mode = 'test')

        self.dataloader.get_dataloader(args, self.get_attrs())
        

    def get_pipe_examples(self, args, mode = None):
        
        if args.backbone == 'bert': 

            from open_intent_discovery.dataloaders.bert_loader import DatasetProcessor
            processor = DatasetProcessor()
            ori_examples = processor.get_examples(self.data_dir, mode)
            if mode == 'train':
                labeled_examples, unlabeled_examples = [], []
                for example in ori_examples:
                    if example.label in self.known_label_list:
                        labeled_examples.append(example)
                    else:
                        example.label = self.unseen_token
                        unlabeled_examples.append(example)

                return labeled_examples, unlabeled_examples

            elif mode == 'eval' or mode == 'test':
                return ori_examples

            






