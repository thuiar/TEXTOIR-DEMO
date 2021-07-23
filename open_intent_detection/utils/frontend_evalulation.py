from collections import defaultdict
import json
import os 

def save_train_results(result_list, args):

    save_dir = os.path.join(args.frontend_result_dir, args.type)  
    save_file_name = 'json_detection_results.json'
    results_path = os.path.join(save_dir, save_file_name)

    train_loss_list = []
    valid_score_list = []

    for elem in result_list:
        train_loss_list.append(elem['train_loss'])
        valid_score_list.append(elem['eval_score'])
    
    record_name = 'detection_' + str(args.dataset) + '_' + str(args.method) + '_' + str(args.log_id)

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:

        json_data = {}
        json_data[record_name] = {}
        json_data[record_name]['Train'] = train_loss_list[:20]
        json_data[record_name]['Valid'] = valid_score_list[:20]

    else:

        with open(results_path, 'r') as f:
            json_data = json.load(f)

        json_data[record_name] = {}
        json_data[record_name]['Train'] = train_loss_list[:20]
        json_data[record_name]['Valid'] = valid_score_list[:20]

    
    json_str = json.dumps(json_data, indent = 4)
    with open(results_path, 'w') as f:
        f.write(json_str)
        

        








