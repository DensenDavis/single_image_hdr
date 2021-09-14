import os
import tensorflow as tf
from shutil import copytree
from datetime import datetime
from config import Configuration
cfg = Configuration()

def clone_checkpoint(ckpt_dir):
    new_ckpt = os.path.join('train_ckpts',os.path.split(cfg.log_dir)[-1])
    if(ckpt_dir != None):
        assert os.path.exists(ckpt_dir)
        copytree(ckpt_dir,new_ckpt)
    ckpt_dir = new_ckpt
    return ckpt_dir

@tf.function()
def mu_tonemapping(hdr_img, norm_val):
    bounded_hdr = tf.math.tanh(hdr_img/norm_val)
    mu_tonemap_output = tf.math.log(1 + cfg.mu * bounded_hdr) / tf.math.log(1 + cfg.mu)
    return mu_tonemap_output