from open_intent_detection.init_parameters import *
from open_intent_detection.dataloader import *
from open_intent_detection.pretrain import *
from open_intent_detection.utils import * 
# from pretrain import *
# from utils import *
# from methods import ADB
# from models import Bert

if __name__ == '__main__':

    #################Open Intent Detection###############3
    print('Open Intent Detection Begin...')
    
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    
    manager = ModelManager(args, data, manager_p.model)
    manager.train(args, data)
    manager.evaluation(args, data, mode="test")  

    # debug(data, manager_p, manager, args)
    print('Training finished!')