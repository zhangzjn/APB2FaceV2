from util.options import get_opt
from util.logger import init_checkpoint, get_logger, show_opt
from data import get_dataset
from trainer import get_trainer
# ---> option
opt = get_opt()
# ---> logging
opt = init_checkpoint(opt)
logger = get_logger(opt)
show_opt(opt, logger)
# ---> data
logger.info('==> Loading data ...')
dataloader = get_dataset(opt)
# ---> model
logger.info('==> Building model ...')
trainer = get_trainer(opt, logger)

for epoch in range(1, 1 + opt.niter + opt.niter_decay):
    trainer.run(dataloader)
trainer.writer.close()
