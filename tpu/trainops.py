import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm
import custom_losses
import utils
from config import Configuration
cfg = Configuration()

class TrainOps():
    def __init__(self, tpu_strategy, dataset, model):
        self.tpu_strategy = tpu_strategy
        self.dataset = dataset
        self.model = model
        with self.tpu_strategy.scope():
            self.mae_loss = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            # self.train_psnr = tf.keras.metrics.Mean(name='train_psnr')
            # self.val_loss = tf.keras.metrics.Mean(name='val_loss')
            self.val_normpsnr = tf.keras.metrics.Mean(name='val_norm_psnr')
            self.val_mupsnr = tf.keras.metrics.Mean(name='val_mu_psnr')
        return

    @tf.function
    def calculate_loss(self, x_input, y_true, y_pred):
        mae_loss = self.mae_loss(y_true,y_pred)
        return tf.nn.compute_average_loss(mae_loss, global_batch_size=cfg.train_batch_size)

    @tf.function
    def train_step(self, data_batch):
        def step_fn(input_batch,gt_batch):
            with tf.GradientTape(persistent=False) as tape:
                output_batch = self.model(input_batch, training=True)
                net_loss = self.calculate_loss(input_batch, gt_batch, output_batch)
            gradients = tape.gradient(net_loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
            # self.train_psnr(tf.image.psnr(dec_outputs, self.gt_batch, max_val=1.0))
        self.tpu_strategy.run(step_fn, args=(data_batch))
        return

    def train_one_epoch(self, epoch):
        self.train_loss.reset_states()
        # self.train_psnr.reset_states()
        pbar = tqdm(self.dataset.train_ds, desc=f'Epoch : {epoch.numpy()}')
        for data_batch in pbar:
            self.train_step(data_batch)
        return

    @tf.function
    def val_step(self, data_batch):
        def step_fn(input_batch,gt_batch):
            hdr_output = self.model(input_batch, training=False)
            align_ratio = 65535.0/tf.math.reduce_max(hdr_output)
            hdr_output = tf.math.round(hdr_output*align_ratio)
            hdr_output = hdr_output/align_ratio
            gt_max_value = tf.math.reduce_max(gt_batch)

            self.val_normpsnr(tf.image.psnr(hdr_output/gt_max_value, gt_batch/gt_max_value, max_val=1.0))
            hdr_output_gc = hdr_output**cfg.gamma
            gt_batch_gc = gt_batch**cfg.gamma
            norm_perc = tfp.stats.percentile(gt_batch,99)
            mu_tonemap_output = utils.mu_tonemapping(hdr_output_gc,norm_perc)
            mu_tonemap_gt = utils.mu_tonemapping(gt_batch_gc,norm_perc)
            self.val_mupsnr(tf.image.psnr(mu_tonemap_output,mu_tonemap_gt, max_val=1.0))
            return hdr_output
        return self.tpu_strategy.run(step_fn, args=(data_batch))

    def generate_display_samples(self, display_batch, output_batch, gt_batch):
        padding_shape = (output_batch.shape[0], output_batch.shape[1], 20, output_batch.shape[3])
        mini_display_batch = np.concatenate((output_batch,np.zeros(padding_shape),gt_batch), axis=2)
        if(type(display_batch)==type(None)):
            display_batch = mini_display_batch
        else:
            display_batch = np.concatenate((display_batch, mini_display_batch), axis=0)
        return display_batch

    def run_validation(self, save_prediction):
        # self.val_loss.reset_states()
        self.val_normpsnr.reset_states()
        self.val_mupsnr.reset_states()
        display_batch = None
        for i,data_batch in enumerate(self.dataset.val_ds, start=1):
            input_batch, gt_batch = data_batch
            output_batch = self.val_step(data_batch)
            if(save_prediction):
                gt_batch = self.tpu_strategy.gather(gt_batch, axis=0)
                output_batch = self.tpu_strategy.gather(output_batch, axis=0)
                display_batch = self.generate_display_samples(display_batch, output_batch, gt_batch)
                if(display_batch.shape[0]>=cfg.display_samples):
                    save_prediction = False
        return display_batch