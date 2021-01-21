import open_intent_detection.init_parameters as init_param
# from dataloader import *
# from pretrain import *
# from utils import * 
def run():
    parser = argparse.ArgumentParser(sys.argv[1:])
    args = parser.parse_args(argv)
    print(args)


if __name__ == '__main__':
    run()

    # #################Open Intent Detection###############3
    # print('Open Intent Detection Begin...')
    # backbone = 'bert

    # parser = init_param.__dict__[backbone]
    # args = parser.parse_args()
    # data = Data(args)

    # if args.pretrain:
    #     manager_p = PretrainBertModelManager(args, data)
    #     manager_p.train(args, data)
    
    # model_name = args.method 
    # from methods.model_name import *

    # manager = ModelManager(args, data, manager_p.model)
    # manager.train(args, data)
    # manager.evaluation(args, data, mode="test")  

    # # debug(data, manager_p, manager, args)
    # print('Training finished!')