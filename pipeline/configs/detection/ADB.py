import sys

class Param():
    
    def __init__(self):

        self.common_param = self.get_common_parameters()
        self.hyper_param = self.get_hyper_parameters()

    def get_common_parameters(self):
        """
        Args:
            type (str): Type for methods.
            dataset (str): The name of the dataset to train selected.
            known_cls_ratio (float): The number of known classes.
            labeled_ratio (float): The ratio of labeled samples in the training set.
            method (str): which method to use.
            train (binary): Whether train the model.
            save_model (binary): Whether save the trained model for open intent detection.
            backbone (str): which backbone to use.
            num_train_epochs (int): The number of training epochs.
            gpu_id (str): Select the GPU id.
            output_dir (str): The output directory where all train data will be written.
            model_dir (str): The output directory where the model predictions and checkpoints will be written.
            result_dir (str): The path to save results.
            results_file_name (str): The file name of all the results.
            save_results (binary): Whether to save final results for open intent detection.
        """
        common_parameters = {

            'type': "open_intent_detection",
            'dataset': 'banking', 
            'known_cls_ratio': 0.75,
            'labeled_ratio': 1.0,
            'method': 'ADB',
            'train': False,
            'save_model': False, 
            'backbone': 'bert',
            'num_train_epochs': 100,
            'gpu_id': '0',
            'output_dir': '/home/sharing/disk2/zhanghanlei/save_data_162/TEXTOIR/outputs', 
            'model_dir': 'models', 
            'results_dir': 'results',
            'results_file_name': 'detection_results.csv',
            'save_results': False
        }

        return common_parameters

    def get_hyper_parameters(self):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs: The training epochs.
            max_seq_len (int): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr_boundary (float): The learning rate of the decision boundary.
            lr (float): The learning rate of backbone.
            loss_fct (str): The loss function for training.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'bert_model': "/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'max_seq_length': None, 
            'freeze_bert_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'activation': 'relu',
            'lr_boundary': 0.05,
            'lr': 2e-5, 
            'loss_fct': 'CrossEntropyLoss',
            'train_batch_size': 128,
            'eval_batch_size': 128,
            'test_batch_size': 128,
            'wait_patient': 10

        }

        return hyper_parameters