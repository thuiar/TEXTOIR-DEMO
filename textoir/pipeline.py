from open_intent_detection.init_parameters import Param as Param_detect
from open_intent_discovery.init_parameters import Param as Param_discover
from run_detect import run as detect 


if __name__ == '__main__':
    print('Parameters Initialization...')
    
    param_detect = Param_detect()
    args_detect = param_detect.args
    param_discover = Param_discover()
    args_discover = param_discover.args

    args_detect.dataset = args_discover.dataset = 'banking'
    args_detect.known_cls_ratio = args_discover.known_cls_ratio = 0.25
    # args_detect.labeled_ratio = args_discover.labeled_ratio = 0.1

    args_detect.method = 'ADB'
    args_discover.method = 'DeepAligned' 


    print('Open Intent Detection Begin...')
    outputs = detect(args_detect)

    print('Open Intent Discovery Begin...')
    
    # run_discovery(args, outputs)