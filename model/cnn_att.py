import tensorflow as tf
from data_helper.sen_id_label import prepocess
from data_helper.embedding import load_word_embedding
from bean.file_path import *
from data_batches.batches import dataset_iterator


class cnn_att_model(object):
    def __init__(self, config, sess):
        # 模型常量定义
        self.n_epochs = config['n_epochs']
        self.filter_sizes = config['filter_sizes']
        self.embedding_size = config['embedding_size']
        self.num_filters = config['num_filters']
        self.num_classes = config['num_classes']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.batch_size = config['batch_size']
        self.sess = sess
        self.l2_loss = tf.constant(0.0)
        self.alpha = None

        self.x_train, self.y_train, self.word_index, self.x_dev, self.y_dev, self.max_len = \
            prepocess(mr_pos_path, mr_neg_path)
        self.x_inputs = tf.placeholder(tf.int32, [None, self.max_len], name='x_inputs')
        self.y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_inputs')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.all_test_acc = []

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv1d(x, W):
        return tf.nn.conv1d(x, W, 1, 'SAME')

    def embedding_layer(self):
        # embedding layer
        with tf.name_scope('embedding'):
            glove_embedding = load_word_embedding(
                word_index=self.word_index,
                file='',
                trimmed_filename=glove_embedding_save,
                load=True,
                dim=300
            )
            glove_w2v = tf.Variable(glove_embedding, dtype=tf.float32, name='glove_w2v')
            batch_embedding = tf.nn.embedding_lookup(glove_w2v, self.x_inputs)
            # sen_inputs_glove = tf.expand_dims(embedded_chars, -1)
        return batch_embedding  # [b_s, max_len, 300]

    def att_embed(self, batch_embedding):
        # 计算注意力
        att_filter = self.weight_variable([5, self.embedding_size, 1])
        att_bias = self.weight_variable([1])
        att_conv = self.conv1d(batch_embedding, att_filter) + att_bias
        self.alpha = tf.nn.sigmoid(att_conv)  # [b_s, max_len, 1]

        # 词向量加权
        batch_embedding_weighted = tf.multiply(self.alpha, batch_embedding)
        return batch_embedding_weighted  # [b_s, max_len, 300]

    def att_res_compute(self, batch_embedding):
        batch_embedding_weighted = self.att_embed(batch_embedding)
        y = tf.reduce_sum(batch_embedding_weighted, 1, keep_dims=True)  # [b_s, 1, 300]

        att_res_filter = self.weight_variable([1, self.embedding_size, 400])
        att_res_bias = self.weight_variable([400])
        att_res_conv = self.conv1d(y, att_res_filter) + att_res_bias
        att_res = tf.nn.tanh(att_res_conv)

        return att_res  # [b_s, 1, 400]

    def cnn_pool(self, batch_embedding):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
            conv = tf.nn.conv2d(
                batch_embedding,
                W, strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv'
            )
            # activation
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # max pooling
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.max_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool'
            )
            pooled_outputs.append(pooled)
        return tf.concat(axis=3, values=pooled_outputs)

    def final_layer(self):
        batch_embedding = self.embedding_layer()
        att_res = self.att_res_compute(batch_embedding)  # [b_s, 1, 400]
        cnn_res = tf.squeeze(self.cnn_pool(tf.expand_dims(batch_embedding, -1)), axis=[1])  # [b_s, 1, 400]

        z_o = tf.concat(axis=2, values=[att_res, cnn_res])  # [b_s, 1, 800]
        w_out = self.weight_variable([1, int(z_o.get_shape()[2]), 256])
        w_out_bias = self.weight_variable([256])
        z_out = tf.nn.tanh(self.conv1d(z_o, w_out) + w_out_bias)  # [b_s, 1, 256]
        w_fc = self.weight_variable([self.batch_size, 256, 250])
        final_res = tf.nn.dropout(tf.matmul(z_out, w_fc), keep_prob=0.75)

        return final_res  # [b_s, 1, 250]

    def softmax_output(self):
        h_drop = tf.squeeze(self.final_layer())

        W = tf.get_variable(
            name='W',
            shape=[h_drop.get_shape()[1], self.num_classes],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')

        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)

        score = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
        prediction = tf.argmax(score, 1, name='prediction')
        return score, prediction

    def opt_op(self):
        score, prediction = self.softmax_output()
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=score,
            labels=self.y_inputs
        )

        loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss  # l2 正则化
        correct_predictions = tf.equal(prediction, tf.argmax(self.y_inputs, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, 'float'),
            name='accuracy'
        )
        optim = tf.train.AdamOptimizer(learning_rate=0.001)  # Adam优化器
        train_op = optim.minimize(loss)  # 使用优化器最小化损失函数

        return train_op, loss, accuracy

    def train(self):
        train_op, loss_op, accuracy_op = self.opt_op()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 划分数据batch
        iterations, next_iterator = dataset_iterator(self.x_train, self.y_train, len(self.x_train))
        for epoch in range(self.n_epochs):
            count = 0
            train_loss = 0
            train_acc = 0
            print("-----------Now we begin the %dth epoch-----------" % (epoch))
            for iter in range(iterations):
                x_train_batch, labels_train_batch = self.sess.run(next_iterator)

                if len(x_train_batch) < self.batch_size:
                    continue

                f_dict = {
                    self.x_inputs: x_train_batch,
                    self.y_inputs: labels_train_batch,
                    self.dropout_keep_prob: 0.5
                }

                _, loss, acc = self.sess.run([train_op, loss_op, accuracy_op], feed_dict=f_dict)
                train_loss = train_loss+loss
                train_acc = train_acc+acc
                count = count + 1
            train_loss = train_loss / count
            train_acc = train_acc / count
            print("-----------After the %dth epoch, the train loss is: %f, the train acc is: %f-----------" % (epoch, train_loss, train_acc))

            # test
            iterations_test, next_iterator_test = dataset_iterator(self.x_dev, self.y_dev, len(self.x_dev))
            self.test(iterations_test, next_iterator_test, epoch, loss_op, accuracy_op)

    def test(self, iterations_test, next_iterator_test, epoch, loss_op, accuracy_op):
        test_loss = 0
        test_acc = 0
        count = 0

        for iter in range(iterations_test):
            x_test_batch, labels_test_batch = self.sess.run(next_iterator_test)

            if len(x_test_batch) < self.batch_size:
                continue

            f_dict = {
                self.x_inputs: x_test_batch,
                self.y_inputs: labels_test_batch,
                self.dropout_keep_prob: 1.0
            }

            count = count + 1
            loss, acc = self.sess.run([loss_op, accuracy_op], feed_dict=f_dict)
            test_loss = test_loss + loss
            test_acc = test_acc+acc

        test_loss = test_loss / count
        test_acc = test_acc / count
        self.all_test_acc.append(test_acc)
        print("-----------After the %dth epoch, the test loss is: %f, the test acc is: %f-----------" % (epoch, test_loss, test_acc))
