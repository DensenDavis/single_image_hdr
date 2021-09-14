import tensorflow as tf
from tensorflow.python.keras import layers
import custom_layers
from tensorflow.python.keras.layers import Conv2D, ReLU
from tensorflow.python.keras.models import Model
from config import Configuration
cfg = Configuration()

class FeedbackBlock(tf.keras.layers.Layer):
  def __init__(self, trainable = True,name=None, **kwargs):
    super(FeedbackBlock,self).__init__(name=name,trainable=trainable, **kwargs)
    self.bottleneck = Conv2D(64, 1, padding='same')
    self.relu = ReLU()
    self.DP1 = custom_layers.DilationPyramid([3,2,1,1], 16)
    self.DP2 = custom_layers.DilationPyramid([3,2,1,1], 16)
    self.concat = layers.Concatenate(axis=-1)
    self.GFF_3x3 = Conv2D(12, 3, padding='same', activation='relu')

  def call(self, inputs, state):
    x_out = self.concat([inputs, state])
    x_out = self.bottleneck(x_out)
    x_out = self.DP1(x_out)
    x_out = self.DP2(x_out)
    x_out = self.relu(x_out)
    x_out = self.GFF_3x3(x_out)
    return x_out

class FHDR(tf.keras.Model):
  def __init__(self, iteration_count, *args, **kwargs):
    super(FHDR,self).__init__(*args, **kwargs)
    self.iteration_count = iteration_count
    self.feb1 = Conv2D(12, kernel_size=3, padding='same', activation='relu')
    self.feb2 = Conv2D(12, kernel_size=3, padding='same', activation='relu')
    self.relu1 = ReLU()
    self.relu2 = ReLU()
    self.feedback = FeedbackBlock()
    self.hrb1 = Conv2D(12, kernel_size=3, padding='same')
    self.hrb2 = Conv2D(3, kernel_size=3, padding='same')

  def call(self,inputs):
    b,h,w,c = inputs.shape
    state_global = tf.zeros([b,h,w,12], dtype=tf.float32)
    outs = []
    feb1 = self.feb1(inputs)
    feb2 = self.feb2(feb1)
    for i in range(self.iteration_count):
      state_global = self.feedback(feb2, state_global)
      FDF = state_global + feb1
      hrb1 = self.relu1(self.hrb1(FDF))
      out = self.hrb2(hrb1)
      out = self.relu2(out)
      outs.append(out)
    return outs


def get_model(input_shape):
  model = FHDR(cfg.rnn_iterations)
  model.build(input_shape)
  return model