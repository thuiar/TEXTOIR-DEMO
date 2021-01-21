from init_parameters import Param
from dataloader import *
from utils import debug

def run():
    print('Parameters Initialization...')
    param = Param()
    args = param.args 

    print('Data Preparation...')
    data = Dataaz(args)


    print('Open Intent Detection Begin...')
    Method = __import__('methods.' + args.method + '.manager')
    Method = Method.__dict__[args.method].manager
    manager = Method.ModelManager(args, data)

    print('Training Begin...')
    manager.train(args, data)
    print('Training Finished...')

    print('Evaluation begin...')
    manager.evaluation(args, data, mode='test')
    print('Evaluation finished...')

    debug(data, manager, args)
    print('Open Intent Detection Finished...')

if __name__ == '__main__':
    run()