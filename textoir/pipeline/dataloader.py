from open_intent_discovery.utils import *
from pipeline.utils import *
from open_intent_discovery.dataloader import Data as Discover
from open_intent_discovery.dataloader import Unsup_Data as Unsup_Discover 
from open_intent_detection.dataloader import Data as Detect
from open_intent_discovery.dataloader import DatasetProcessor

class Data_detect(Detect):

    def __init__(self, args):
        
        super(Data_detect, self).__init__(args)

        tmp_all_label_list = self.all_label_list
        self.all_label_list = copy.copy(self.known_label_list)
        
        for label in tmp_all_label_list:
            if label not in self.known_label_list:
                self.all_label_list.append(label)

        self.test_examples = self.get_test_examples()
        self.test_dataloader = self.get_loader(self.test_examples, args, self.all_label_list, 'test')

    def get_test_examples(self):
        
        processor = DatasetProcessor()
        ori_examples = processor.get_examples(self.data_dir, 'test')
        

        return ori_examples

class Data_discover(Discover):

    def __init__(self, args):
        
        super(Data_discover, self).__init__(args)

        self.data_dir = args.pipe_results_path
        self.known_label_list = list(load_npy(args.pipe_results_path, 'known_labels.npy'))
        self.all_label_list = list(load_npy(args.pipe_results_path, 'all_labels.npy'))

        self.n_known_cls = len(self.known_label_list)
        self.num_labels = int((len(self.all_label_list)) * args.cluster_num_factor)   
        
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        
        self.all_label_list.append(self.unseen_token)
        self.unseen_token_id = len(self.known_label_list)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_pipe_examples(mode = 'train')
        self.eval_examples = self.get_pipe_examples(mode = 'eval')
        self.test_examples = self.get_pipe_examples(mode = 'test')
        print('labeled_examples', len(self.train_labeled_examples))
        print('unlabeled_examples', len(self.train_unlabeled_examples))

        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train_l')
        self.train_unlabeled_dataloader = self.get_loader(self.train_unlabeled_examples, args, 'train_u')
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataloader = self.get_semi_loader(self.input_ids, self.input_mask, self.segment_ids, self.label_ids, args)
        
        self.train_dataloader = self.train_semi_dataloader
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_pipe_examples(self, mode=None):
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

class Unsup_Data_Discover(Unsup_Discover):

    def __init__(self, args):
        
        super(Unsup_Data_Discover, self).__init__(args)
        self.data_dir = args.pipe_results_path
        self.known_label_list = list(load_npy(args.pipe_results_path, 'known_labels.npy'))
        self.all_label_list = list(load_npy(args.pipe_results_path, 'all_labels.npy'))
        self.n_known_cls = len(self.known_label_list)
        self.num_labels_ = int((len(self.all_label_list)) * args.cluster_num_factor)   
        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.all_label_list.append(self.unseen_token)
        #### transform labels
        self.le.classes_ = self.all_label_list
        self.all_data['y_true'] = self.le.transform(self.all_data['label'])
        ####

        self.unseen_token_id = len(self.known_label_list)
        self.test_data_length = self.read_csv_by_mode(mode = 'test')
        self.X_train, self.X_test, self.y_train, self.y_test, self.df_train, self.df_test = self.get_train_test_by_file()

        print('train_examples', len(self.X_train))
        print('test_examples', len(self.X_test))
        self.num_labels = int(len(set(self.y_train)) * args.cluster_num_factor)
        print("num labels: {}".format(self.num_labels))
        

    def read_csv_by_mode(self, mode='train'):
        if mode == 'train':
            train = pd.read_csv(os.path.join(self.data_dir, 'train.tsv'), sep = '\t')
            dev = pd.read_csv(os.path.join(self.data_dir, 'dev.tsv'), sep = '\t')
            l_train = [[x,y] for x,y in zip(train['text'], train['label'])]
            l_dev = [[x,y] for x,y in zip(dev['text'], dev['label'])]
            l_all = l_train + l_dev
        elif mode == 'test':
            test = pd.read_csv(os.path.join(self.data_dir, 'test.tsv'), sep = '\t')
            l_all = [[x,y] for x,y in zip(test['text'], test['label'])]

        return len(l_all)

    def get_train_test_by_file(self):

        df_train, df_test = self.all_data.iloc[:-self.test_data_length, :], self.all_data.iloc[-self.test_data_length:, :]
        X_train = self.sequences_pad[df_train.index]
        X_test = self.sequences_pad[df_test.index]
        y_train = df_train.y_true.values
        y_test = df_test.y_true.values

        return X_train, X_test, y_train, y_test, df_train, df_test