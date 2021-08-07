import os

def discover_centers(args, data, outputs):
    
    model_dir, output_file_dir, _ = set_path(args)

    predictions = list([data.all_label_list[idx] for idx in outputs[0]]) 
    true_labels = list([data.all_label_list[idx] for idx in outputs[1]]) 
    feats = outputs[2]
    reduce_feats = TSNE_reduce_feats(feats, 2)

    reduce_centers = []
    labels = np.unique(outputs[1])
    for label in labels:
        print(data.all_label_list[label])
        pos = list(np.where(outputs[1] == label)[0])
        center = np.mean(reduce_feats[pos], axis = 0)
        print('center', center)
        center = [round(float(x), 2) for x in center]
        reduce_centers.append(center)

    print(reduce_centers)
    # reduce_centers = [round(x, 2) for x in reduce_centers]
    all_dict = {}
    static_dir = os.path.join(args.frontend_dir, args.type)
    draw_center_r_path = os.path.join(static_dir, args.method + '_analysis.json') 
    
    known_centers = []
    open_centers = []
    for idx, center in enumerate(reduce_centers):
        label = data.all_label_list[idx]
        if label in data.known_label_list:
            point = center + [label]
            known_centers.append(point)
        else:
            point = center + [label]
            open_centers.append(point)

    center_dict = {}
    center_dict['Known Intent Centers'] = known_centers
    center_dict['Open Intent Centers'] = open_centers
    name = args.method + '_' +  args.dataset 
    all_dict[name] = center_dict
    json_add(all_dict, draw_center_r_path)
    
class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            SAE_feats_path (directory): The path for pre-trained stacked auto-encoder features.
            num_train_epochs_SAE (int): The number of epochs for training stacked auto-encoder.
            num_train_epochs_DEC (int): The number of epochs for training DEC model.
            feat_dim (int): The feature dimension.
            update_interval (int): The number of intervals between contiguous updates.
            lr (float): The learning rate for training DCN.
            momentum (float): The momentum value of SGD optimizer.
            tol (float): The tolerance threshold to stop training for DCN.
            model_name (str): The name of the DCN model (saved in the format of keras).
        """
        hyper_parameters = {
            'SAE_feats_path': os.path.join('_'.join([str(x) for x in ['SAE', args.dataset, 'sae', str(args.seed)]]), 'models', 'SAE.h5'),
            'num_train_epochs_SAE': 5000,
            'num_train_epochs_DEC': 12000,
            'feat_dim': 2000,
            'update_interval': 100,
            'batch_size': 256,
            'lr': 0.001,
            'momentum': 0.9,
            'tol': 0.01,
            'model_name': 'DEC.h5'
        }

        return hyper_parameters