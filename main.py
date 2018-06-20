import argparse

from model import Reader
from config import Config
from prepro import prepro
from train import Trainer

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help="which mode", choices=config.CMD)
parser.add_argument('-epoch', type=int, help="training epoches",default=config.training.epoch)
parser.add_argument('-lr', type=float, help="learning rate",default=config.training.lr)
parser.add_argument('-batch_size', type=int, help="batch size",default=config.training.batch_size)
parser.add_argument('-l2', type=float, help="l2_regulation",default=config.training.l2)

args = parser.parse_args()

config.training.epoch = args.epoch
config.training.lr = args.lr
config.training.batch_size = args.batch_size
config.training.l2 = args.l2

if args.mode == 'prepro':
    prepro(config)
elif args.mode == 'train':
    trainer = Trainer(config)
    trainer.start()
elif args.mode == 'evaluate':
    pass
elif args.mode == 'interactive':
    pass
else:
    raise RuntimeError("Mode %s is undefined." % args.mode)