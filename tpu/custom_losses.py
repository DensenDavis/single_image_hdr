import tensorflow as tf
import tensorflow_probability as tfp
import utils
from config import Configuration
cfg = Configuration()

class CR_Loss(tf.keras.losses.Loss):
    def __init__(self,name='CR_Loss', **kwargs):
        super(CR_Loss, self).__init__()
        self.vgg_model = self.get_vgg19_model()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mul_list = tf.constant([1/32,1/16,1/8,1/4,1], dtype=tf.float32)
        self.feature_loss_pos = tf.Variable([0,0,0,0,0], trainable=False, dtype=tf.float32)
        self.feature_loss_neg = tf.Variable([0,0,0,0,0], trainable=False, dtype=tf.float32)

    def get_vgg19_model(self):
        vgg = tf.keras.applications.VGG19(input_shape=(None, None, 3), include_top=False)
        layer_names = ["block1_pool", "block2_pool", "block3_pool","block4_pool", "block5_pool"]
        return tf.keras.Model(vgg.input, [vgg.get_layer(layer_name).output for layer_name in layer_names])

    def get_vgg_features(self, img):
        return self.vgg_model(img)

    def calc_loss(self, x_input, y_true, y_pred):
        features_true = self.get_vgg_features(y_true)
        features_pred = self.get_vgg_features(y_pred)
        features_input = self.get_vgg_features(x_input)

        for i in range(len(features_true)):
            self.feature_loss_pos[i].assign(self.mae(features_true[i], features_pred[i]))
            self.feature_loss_neg[i].assign(self.mae(features_input[i], features_pred[i]))
        
        cr_loss = self.mul_list*(self.feature_loss_pos/self.feature_loss_neg)
        return tf.reduce_sum(cr_loss)

class MU_Tone_Loss(tf.keras.losses.Loss):
    def __init__(self,name='MU_Tone_Loss', mu=cfg.mu,gamma=cfg.gamma,percentile=99,**kwargs):
        super(MU_Tone_Loss, self).__init__()
        self.mu = mu
        self.gamma = gamma
        self.percentile = percentile

    def call(self, y_true,y_pred):
        pred = y_pred**self.gamma
        gt = y_true**self.gamma
        gt_reshape = tf.reshape(gt, [-1, cfg.train_img_shape[0]*cfg.train_img_shape[1]*3])
        norm_perc = tfp.stats.percentile(gt_reshape,self.percentile)
        norm_perc = tf.reshape(norm_perc, [-1,1])
        mu_tonemap_output = utils.mu_tonemapping(pred,norm_perc)
        mu_tonemap_gt = utils.mu_tonemapping(gt,norm_perc)
        return tf.reduce_mean((mu_tonemap_output - mu_tonemap_gt)**2)