import os
import sys
import time
import logging


def init_checkpoint(opt):
    if not os.path.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    if opt.resume:
        opt.name_dir = opt.resume_name
        opt.logdir = '{}/{}'.format(opt.checkpoint, opt.name_dir)
    else:
        opt.name_dir = '{}-{}'.format(opt.name, time.strftime("%Y%m%d-%H%M%S"))
        opt.logdir = '{}/{}'.format(opt.checkpoint, opt.name_dir)
        os.mkdir(opt.logdir)
    return opt


def get_logger(opt):
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler('{}/log_{}.txt'.format(opt.logdir, opt.mode))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    return logger


def show_opt(opt, logger):
    for key, val in vars(opt).items():
        if isinstance(val, list):
            val = [str(v) for v in val]
            val = ','.join(val)
        if val is None:
            val = 'None'
        logger.info('{:>20} : {:<50}'.format(key, val))


