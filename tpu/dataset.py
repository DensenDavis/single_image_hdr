import tensorflow as tf
import math
import glob
import numpy as np
import cv2
from tqdm import tqdm
from config import Configuration
autotune = tf.data.experimental.AUTOTUNE
cfg = Configuration()

class Dataset():
    def __init__(self, tpu_strategy):
        self.train_gt_files = self.read_images(sorted(glob.glob(cfg.train_gt_path)), "train gt")
        self.train_gt_ar_files = self.read_align_ratios(sorted(glob.glob(cfg.train_gt_alignratio_path)))
        self.train_input_files = self.read_images(sorted(glob.glob(cfg.train_input_path)),"train input")
        self.num_train_imgs = self.train_input_files.shape[0]
        self.num_train_batches = self.num_train_imgs/cfg.train_batch_size
        self.val_gt_files = self.read_images(sorted(glob.glob(cfg.val_gt_path)),"val gt")
        self.val_gt_ar_files = self.read_align_ratios(sorted(glob.glob(cfg.val_gt_alignratio_path)))
        self.val_input_files = self.read_images(sorted(glob.glob(cfg.val_input_path)),"val input")
        self.num_val_images = self.val_input_files.shape[0]
        self.train_ds = tpu_strategy.experimental_distribute_dataset(self.get_train_data())
        self.val_ds = tpu_strategy.experimental_distribute_dataset(self.get_val_data())

    def extract_chunks(self, lst, chunk_size=10):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def read_images(self, filenames, image_type):
        files_tf = None
        chunk_size = 50
        n_chunks = math.ceil(len(filenames)//chunk_size)
        batches = self.extract_chunks(filenames, chunk_size)
        pbar = tqdm(batches, desc=f'Reading {image_type} images -> chunk size = 50', total=n_chunks)
        for batch in pbar:
            file_raw = []
            for i in batch:
                with open(i, 'rb') as f:
                    file_raw.append(f.read())
            files_np =  np.asarray(file_raw)
            if files_tf is None:
                files_tf = tf.convert_to_tensor(files_np)
            else:
                files_tf = tf.concat([files_tf,tf.convert_to_tensor(files_np)], axis=0)
        print(f'Extraction complete. Total {image_type} files = {files_tf.shape[0]}')
        return files_tf

    def read_align_ratios(self, filenames):
        align_ratios = []
        for i in filenames:
            align_ratios.append(np.load(i).astype(np.float32))
        return tf.convert_to_tensor(align_ratios)

    def decode_images(self, input_img, target_img, gt_ds_alignratio):
        input_img = tf.image.decode_image(input_img, dtype=tf.dtypes.uint8)
        target_img = tf.image.decode_image(target_img, dtype=tf.dtypes.uint16)
        return input_img, target_img, gt_ds_alignratio

    def create_pair(self, input_img, target_img, gt_ds_alignratio):
        input_img = tf.cast(input_img, tf.dtypes.float32)
        input_img = input_img/255.0
        target_img = tf.cast(target_img, tf.dtypes.float32)
        target_img = target_img/gt_ds_alignratio
        return tf.concat([input_img, target_img], axis=-1)

    def create_train_crop(self, img_pair):
        img_patch = tf.image.random_crop(img_pair, [cfg.train_img_shape[0], cfg.train_img_shape[1], cfg.train_img_shape[-1]*2])
        return img_patch

    def create_val_crop(self, img_pair):
        img_pair = tf.image.crop_to_bounding_box(img_pair, 0, 0, cfg.val_img_shape[0], cfg.val_img_shape[1])
        return img_pair

    def split_train_pair(self, image_pair):
        return image_pair[:,:,:cfg.train_img_shape[-1]],image_pair[:,:,cfg.train_img_shape[-1]:]

    def split_val_pair(self, image_pair):
        return image_pair[:,:,:cfg.val_img_shape[-1]],image_pair[:,:,cfg.val_img_shape[-1]:]

    def train_augmentation(self, img_pair):
        img_pair = tf.image.random_flip_up_down(img_pair)
        img_pair = tf.image.random_flip_left_right(img_pair)
        img_pair = tf.image.rot90(img_pair, k=tf.random.uniform([], maxval=5, dtype=tf.int32))
        return img_pair

    def val_augmentation(self, img_pair):
        imgs_ud_flip = tf.image.flip_up_down(img_pair)
        imgs_lr_flip = tf.image.flip_left_right(img_pair)
        # img_pair_3 = tf.image.rot90(img_pair, k=tf.random.uniform([], maxval=5, dtype=tf.int32))
        img_pair = tf.concat([img_pair, imgs_ud_flip, imgs_lr_flip], axis=0)
        return img_pair

    def get_train_data(self):
        input_ds = tf.data.Dataset.from_tensor_slices(self.train_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.train_gt_files)
        gt_ds_alignratio = tf.data.Dataset.from_tensor_slices(self.train_gt_ar_files)
        ds = tf.data.Dataset.zip((input_ds, gt_ds, gt_ds_alignratio))
        ds = ds.shuffle(buffer_size=50, reshuffle_each_iteration=True)
        ds = ds.map(self.decode_images, num_parallel_calls=autotune)
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_train_crop, num_parallel_calls=autotune)
        if cfg.train_augmentation:
            ds = ds.map(self.train_augmentation, num_parallel_calls=autotune)
        ds = ds.map(self.split_train_pair, num_parallel_calls=autotune)
        ds = ds.batch(cfg.train_batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=autotune)
        return ds

    def get_val_data(self):
        input_ds = tf.data.Dataset.from_tensor_slices(self.val_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.val_gt_files)
        gt_ds_alignratio = tf.data.Dataset.from_tensor_slices(self.val_gt_ar_files)
        ds = tf.data.Dataset.zip((input_ds, gt_ds, gt_ds_alignratio))
        # ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
        ds = ds.map(self.decode_images, num_parallel_calls=autotune)
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_val_crop, num_parallel_calls=autotune)
        if cfg.val_augmentation:
            ds = ds.map(self.val_augmentation, num_parallel_calls=autotune)
            ds = ds.unbatch()
            ds = ds.shuffle(buffer_size=20, reshuffle_each_iteration=True)
            ds = ds.batch(cfg.val_batch_size)
        ds = ds.map(self.split_val_pair, num_parallel_calls=autotune)
        ds = ds.batch(cfg.val_batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=autotune)
        return ds