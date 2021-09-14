import os
import tensorflow as tf
from model import get_model
from config import Configuration
import cv2
import time
import numpy as np
cfg = Configuration()

input_path = '/content/ds/val'
save_path = './res'
os.makedirs(save_path, exist_ok=True)
fileNames = sorted(os.listdir(input_path))

model = get_model([None, None, 3])
ckpt = tf.train.Checkpoint(model = model)
ckpt_dir = 'train_ckpts/xxxx-xxxx/best'

chkpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
ckpt.restore(chkpt_manager.latest_checkpoint)
_ = model(tf.ones([1,1060,1900,3])) # dummy input to build the graph

avg_time = tf.keras.metrics.Mean()
for i in range(len(fileNames)):
    print(fileNames[i])
    input_image = tf.io.read_file(input_path + '/'+ fileNames[i])
    input_image = tf.image.decode_image(input_image, dtype=tf.dtypes.float32)

    time_init = time.time()
    hdr_output = model(tf.expand_dims(input_image,0), training=False)[0]
    avg_time(time.time()-time_init)

    align_ratio = (2 ** 16 - 1)/tf.math.reduce_max(hdr_output)
    hdr_output = tf.math.round(hdr_output*align_ratio)
    hdr_output = tf.cast(hdr_output, tf.dtypes.uint16)

    header_name = save_path+'/'+f'{fileNames[i][:4]}'
    np.save(header_name+'_alignexposure.npy', align_ratio.numpy())
    cv2.imwrite(header_name+'.png', cv2.cvtColor(hdr_output.numpy(), cv2.COLOR_RGB2BGR))

print('inference completed')
print(f'runtime/image = {avg_time.result()}s')
pass
