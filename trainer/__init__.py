import importlib
import torch


def get_trainer(opt, logger):
    trainer_filename = 'trainer.{}_trainer'.format(opt.trainer)
    trainerlib = importlib.import_module(trainer_filename)
    trainer = trainerlib.Trainer_
    trainer_use = trainer(opt, logger)
    return trainer_use