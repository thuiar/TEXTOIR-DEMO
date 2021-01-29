import os
import numpy as np
import pandas as pd
import torch
import random
import csv
import sys
import copy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class Data:
    
    
    def __init__(self, args, mode='detect'):
        set_seed(args.seed)
        max_seq_lengths = {'oos':30, 'stackoverflow':45,'banking':55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        self.mode = mode
        processor = DatasetProcessor()
        
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'

        if self.mode == 'detect':

            self.data_dir = os.path.join(args.data_dir, args.dataset)

            all_label_list = processor.get_labels(self.data_dir)
            self.n_known_cls = round(len(all_label_list) * args.known_cls_ratio)
            self.known_label_list = list(np.random.choice(np.array(all_label_list), self.n_known_cls, replace=False))    
        
            self.all_label_list = copy.copy(self.known_label_list)
            for label in all_label_list:
                if label not in self.known_label_list:
                    self.all_label_list.append(label)
            
            print('len_labels',len(self.all_label_list))
            self.label_list = self.known_label_list + [self.unseen_token]
            self.all_label_list.append(self.unseen_token)

            self.save_npy(np.array(self.all_label_list), args.pipe_results_path, 'labels.npy')
            self.save_npy(np.array(self.known_label_list), args.pipe_results_path, 'known_labels.npy')

            self.unseen_token_id = len(self.known_label_list)
            self.num_labels = len(self.known_label_list)

            self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
            print('num_labeled_samples',len(self.train_labeled_examples))
            print('num_unlabeled_samples',len(self.train_unlabeled_examples))
            self.train_examples = self.train_labeled_examples

            self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, self.known_label_list, 'train')
            self.train_unlabeled_dataloader = self.get_loader(self.train_unlabeled_examples, args, self.label_list, 'train')
            self.train_dataloader = self.train_labeled_dataloader
            
            self.eval_examples = self.get_examples(processor, args, 'eval')
            self.test_examples = self.get_examples(processor, args, 'test')
            self.test_true_examples = self.get_examples(processor, args, 'test', test_true=True)

            self.eval_dataloader = self.get_loader(self.eval_examples, args, self.known_label_list ,'eval')
            self.test_dataloader = self.get_loader(self.test_examples, args, self.label_list, 'test')
            self.test_true_dataloader = self.get_loader(self.test_true_examples, args, self.all_label_list, 'test')

        elif self.mode == 'discover':

            self.data_dir = args.data_dir
            self.all_label_list = list(self.load_npy(args.pipe_results_path, 'labels.npy'))
            self.known_label_list = list(self.load_npy(args.pipe_results_path, 'known_labels.npy'))
            self.n_known_cls = len(self.known_label_list)
            self.num_labels = int((len(self.all_label_list)) * args.cluster_num_factor)

            self.all_label_list.append(self.unseen_token)
            self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
            print('num_labeled_samples',len(self.train_labeled_examples))
            print('num_unlabeled_samples',len(self.train_unlabeled_examples))
            
            # for example in train_labeled_examples:
            #     print('text',example.text_a)
            #     print('label',example.label)
            #     if example.label in self.known_label_list:
            #         print('yes')
            #     else:
            #         print('no')

            self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, self.known_label_list, 'train')
            self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
            self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, args)

            self.eval_examples = self.get_examples(processor, args, 'eval')
            self.test_examples = self.get_examples(processor, args, 'test')

            self.eval_dataloader = self.get_loader(self.eval_examples, args, self.known_label_list ,'eval')
            self.test_dataloader = self.get_loader(self.test_examples, args, self.all_label_list, 'test')

    def get_examples(self, processor, args, mode = 'train', test_true = False):

        ori_examples = processor.get_examples(self.data_dir, mode)
        
        if mode == 'train':
            if self.mode == 'detect':
                labeled_examples, unlabeled_examples = [], []
                train_labels = np.array([example.label for example in ori_examples])
                train_labeled_ids, train_unlabeled_ids = [], []
                
                for label in self.known_label_list:
                    num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                    pos = list(np.where(train_labels == label)[0])                
                    train_labeled_ids.extend(random.sample(pos, num))

                for idx, example in enumerate(ori_examples):
                    if idx in train_labeled_ids:
                        labeled_examples.append(example)
                    else:
                        example.label = self.unseen_token
                        unlabeled_examples.append(example)
                return labeled_examples, unlabeled_examples

            elif self.mode == 'discover':
                labeled_examples, unlabeled_examples = [], []
                for example in ori_examples:
                    if example.label in self.known_label_list:
                        labeled_examples.append(example)
                    else:
                        example.label = self.unseen_token
                        unlabeled_examples.append(example)
                return labeled_examples, unlabeled_examples

        elif mode == 'eval':
            examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    examples.append(example)        
            return examples

        elif mode == 'test':
            
            examples = []
            if self.mode == 'detect' and not test_true:
                for example in ori_examples:
                    if (example.label in self.label_list) and (example.label is not self.unseen_token):
                        examples.append(example)
                    else:
                        example.label = self.unseen_token
                        examples.append(example) 
            else:     
                for example in ori_examples:
                    examples.append(example)        
                return examples  

            return examples            
    
    def get_loader(self, examples, args, label_list, mode = 'train'):

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size) 
        
        return dataloader

    def get_semi(self, labeled_examples, unlabeled_examples, args):
        
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length, tokenizer)
        unlabeled_features = convert_examples_to_features(unlabeled_examples, self.all_label_list, args.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)      

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)      

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])

        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):

        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

        return semi_dataloader

    def save_npy(self, npy_file, path, file_name):
        npy_path = os.path.join(path, file_name)
        np.save(npy_path, npy_file)
    
    def load_npy(self, path, file_name):
        npy_path = os.path.join(path, file_name)
        npy_file = np.load(npy_path)
        return npy_file

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        if os.path.exists(os.path.join(data_dir, 'labels.npy')):
            labels = np.load(os.path.join(data_dir, 'labels.npy'))
        else:
            test = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t")
            labels = np.unique(np.array(test['label']))
            
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()