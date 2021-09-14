import os
import numpy as np
import wandb
import tensorflow as tf
from model import get_model
from config import Configuration
from dataset import Dataset
from train_loop import TrainLoop
from utils import clone_checkpoint
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as lr_decay
cfg = Configuration()


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
tpu_strategy = tf.distribute.TPUStrategy(resolver)

dataset = Dataset(tpu_strategy)
lr_schedule = lr_decay(
    boundaries=[i*dataset.num_train_batches for i in cfg.lr_boundaries],
    values=cfg.lr_values)

with tpu_strategy.scope():
  model = get_model([None, None, 3])
  model.optimizer = tf.keras.optimizers.Adam(lr_schedule)

train_obj = TrainOps(tpu_strategy,dataset, model)
wandb.init(project='hdr_tpu')

ckpt = tf.train.Checkpoint(
    model = train_obj.model,
    epoch = tf.Variable(0, dtype=tf.dtypes.int64),
    max_psnr = tf.Variable(0.0))

ckpt_dir = clone_checkpoint(cfg.ckpt_dir)
chkpt_best = os.path.join(ckpt_dir,'best')
chkpt_best = tf.train.CheckpointManager(ckpt, chkpt_best, max_to_keep=1, checkpoint_name='ckpt')
chkpt_last = os.path.join(ckpt_dir,'last')
chkpt_last = tf.train.CheckpointManager(ckpt, chkpt_last, max_to_keep=1, checkpoint_name='ckpt')
ckpt_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")

if cfg.train_mode == 'best':
    ckpt.restore(chkpt_best.latest_checkpoint, options=ckpt_options)
elif cfg.train_mode == 'last':
    ckpt.restore(chkpt_last.latest_checkpoint, options=ckpt_options)
else:
    raise Exception('Error! invalid training mode, please check the config file.')

print(f"Initiating training from epoch {ckpt.epoch.numpy()}")
print(f'best_psnr = {ckpt.max_psnr.numpy()}')

while(ckpt.epoch<cfg.n_epochs):
    ckpt.epoch.assign_add(1)
    train_obj.train_one_epoch(ckpt.epoch)
    if ckpt.epoch%cfg.val_freq == 0:
        save_prediction = ckpt.epoch%cfg.display_frequency==0
        display_batch = train_obj.run_validation(save_prediction)
        wandb.log({"val norm psnr": train_obj.val_normpsnr.result().numpy(),
                    "val mu psnr":train_obj.val_mupsnr.result().numpy()}
                )
        wandb.run.summary["epoch"] = ckpt.epoch.numpy()
        if(save_prediction):
            wandb.log({"val_images":wandb.Image(display_batch)})
        if ckpt.max_psnr<=train_obj.val_mupsnr.result():
            ckpt.max_psnr.assign(train_obj.val_mupsnr.result())
            chkpt_best.save(checkpoint_number=1, options=ckpt_options)
            wandb.run.summary["best_mupsnr"] = train_obj.val_mupsnr.result().numpy()
    chkpt_last.save(checkpoint_number=1, options=ckpt_options)
    print(f'psnr : best/last = {ckpt.max_psnr.numpy()}/{train_obj.val_mupsnr.result().numpy()}')