import tensorflow as tf
from bean.configuration import config_cnn, config_rnn, config_cnn_att
from model import cnn, cnn_att, lstm, lstm_att, gru, gru_att


def modle_train(dl_model, c):
    with tf.Session() as sess:
        model = dl_model(c, sess)
        model.train()
        mean_test_acc = sum(model.all_test_acc) / len(model.all_test_acc)
        max_test_acc = max(model.all_test_acc)
        return mean_test_acc, max_test_acc


mean_test_acc, max_test_acc = modle_train(cnn.cnn_model, config_cnn_att)
# mean_test_acc, max_test_acc = modle_train(cnn_att.cnn_att_model, config)
# mean_test_acc, max_test_acc = modle_train(lstm.lstm_model, config_lstm)
# mean_test_acc, max_test_acc = modle_train(lstm_att.lstm_att_model, config_lstm)
# mean_test_acc, max_test_acc = modle_train(gru.gru_model, config_lstm)
# mean_test_acc, max_test_acc = modle_train(gru_att.gru_att_model, config_rnn)
print(mean_test_acc, max_test_acc)
