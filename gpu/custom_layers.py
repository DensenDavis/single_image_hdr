import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, ReLU
import tensorflow_addons as tfa


class CA_Block(tf.keras.layers.Layer):
  """[This Layer performs the channelwise attention on the input featuremap.
      Ref: https://arxiv.org/abs/1911.07559]

  Args:
      filters: Depth of feature maps in convolutional layers
  """
  def __init__(self,filters,trainable = True,name=None, **kwargs):
    super(CA_Block,self).__init__(name=name,trainable=trainable, **kwargs)
    self.avg_pool = tfa.layers.AdaptiveAveragePooling2D((1,1))
    self.body = [layers.Conv2D(filters//8,kernel_size=1,padding='same',activation='relu'),\
    layers.Conv2D(filters,kernel_size=1,padding='same',activation='sigmoid')]
    self.mul = layers.Multiply()

  def build(self,input_shapes):
    _,_,channels = input_shapes[1:]
    self.layer_body = tf.keras.Sequential([layers.Input([1,1,channels]),*self.body])

  def call(self,inputs):
    x_pool = self.avg_pool(inputs)
    x_feat = self.layer_body(x_pool)
    out = self.mul([inputs,x_feat])
    return out


class PA_Block(tf.keras.layers.Layer):
  """[This Layer performs the pixelwise attention on the input featuremap.
      Ref: https://arxiv.org/abs/1911.07559]

  Args:
      filters: Depth of feature maps in convolutional layers
  """
  def __init__(self,filters,trainable = True,name=None, **kwargs):
    super(PA_Block,self).__init__(name=name,trainable=trainable, **kwargs)
    self.body = [layers.Conv2D(filters//8,kernel_size=1,padding='same',activation='relu'),\
    layers.Conv2D(1,kernel_size=1,padding='same',activation='sigmoid')]
    self.mul = layers.Multiply()

  def build(self,input_shapes):
    h,w,channels = input_shapes[1:]
    self.layer_body = tf.keras.Sequential([layers.Input([h,w,channels]),*self.body])

  def call(self,inputs):
    x_feat = self.layer_body(inputs)
    out = self.mul([inputs,x_feat])
    return out


class FA_Block(tf.keras.layers.Layer):
  """[This Layer performs the feature attention on the input featuremap.
      Ref: https://arxiv.org/abs/1911.07559]

  Args:
      filters: Depth of feature maps in convolutional layers
      kernel: Kernel size of conv layers
  """
  def __init__(self,filters,kernel,trainable = True,name=None, **kwargs):
    super(FA_Block,self).__init__(name=name,trainable=trainable, **kwargs)
    self.cnv_1 = Conv2D(filters,kernel_size=kernel,padding='same',activation='relu')
    self.add = layers.Add()
    self.conv_2 = layers.Conv2D(filters,kernel_size=kernel,padding='same',activation='relu')
    self.body = [CA_Block(filters),PA_Block(filters)]

  def build(self,input_shapes):
    h,w,channels = input_shapes[1:]
    self.layer_body = tf.keras.Sequential([layers.Input([h,w,channels]),*self.body])
    self.built = True
  
  def call(self,inputs):
    x_feat = self.cnv_1(inputs)
    x_feat = self.add([inputs,x_feat])
    x_feat = self.conv_2(x_feat)
    x_feat = self.layer_body(x_feat)
    out = self.add([inputs,x_feat])
    return out

class DilationPyramid(tf.keras.layers.Layer):
  def __init__(self, dilation_rates = [3,2,1,1], n_filters=32):
    super(DilationPyramid, self).__init__()
    self.convs = [Conv2D(n_filters,3,1,'same',dilation_rate = i) for i in dilation_rates]
    self.relus = [ReLU() for i in range(len(dilation_rates))]
    self.concats = [layers.Concatenate(axis=-1) for i in range(len(dilation_rates))]
    self.add = layers.Add()
  
  def build(self, input_shape):
    self.bottleneck = Conv2D(input_shape[-1],1,1,'same')

  def call(self,x_in):
    x_in_input = x_in
    in_channels = x_in.shape[-1]
    for i in range(len(self.convs)):
        x_out = self.relus[i](x_in)
        x_out = self.convs[i](x_out)
        x_in = self.concats[i]([x_in,x_out])
    x_out = self.bottleneck(x_in) # bottleneck
    x_out = self.add([x_in_input,x_out])
    return x_out