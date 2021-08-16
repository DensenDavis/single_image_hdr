import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.keras import layers
import custom_layers
from tensorflow.python.keras.layers import Conv2D, ReLU
from tensorflow.python.keras.models import Model


class FHDR(tf.keras.Model):
  def __init__(self, iteration_count, *args, **kwargs):
    super(FHDR,self).__init__(*args, **kwargs)
    self.iteration_count = iteration_count
    self.feb1 = Conv2D(12, kernel_size=3, padding='same', activation='relu')
    self.feb2 = Conv2D(12, kernel_size=3, padding='same', activation='relu')
    self.relu1 = ReLU()
    self.relu2 = ReLU()
    self.fa_block = FA_Block(12,3)
    self.hrb1 = Conv2D(12, kernel_size=3, padding='same')
    self.hrb2 = Conv2D(3, kernel_size=3, padding='same')
    self.state_init = tf.constant((tf.zeros([2,128,128,12], dtype=tf.float32)))

  def call(self,inputs):
    # self.fa_block.reset_states()
    feb1 = self.feb1(inputs)
    feb2 = self.feb2(feb1)
    b,h,w,c = inputs.shape
    state = tf.zeros([b,h,w,12], dtype=tf.float32)
    tf.print(state.shape)
    # state = self.state_init
    outs = []
    for i in range(self.iteration_count):
      fb_out, state = self.fa_block(feb2, state)
      FDF = fb_out + feb1
      hrb1 = self.relu1(self.hrb1(FDF))
      out = self.hrb2(hrb1)
      out = self.relu2(out)
      outs.append(out)
    return outs


model = FHDR(3)
model.build([2,128,128,3])
print(model.summary())
tf.keras.utils.plot_model(model, 'mo.png',show_shapes=True, show_layer_names=True)

outs = model(tf.random.normal((2,128,128,3), mean=0.5, stddev=0.5, seed=1), training=False)
outs2 = model(tf.random.normal((2,128,128,3), mean=0.5, stddev=0.5, seed=10), training=False)
outs3 = model(tf.random.normal((2,128,128,3), mean=0.5, stddev=0.5, seed=3), training=False)