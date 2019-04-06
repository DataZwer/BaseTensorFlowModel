from model.cnn_att import cnn_att_model
import tensorflow as tf
from bean.configuration import config, config_cnn_att


with tf.Session() as sess:
    model = cnn_att_model(config_cnn_att, sess)
    final_res = model.final_layer()
