import tensorflow as tf
from data_helper.sen_id_label import prepocess
from data_helper.embedding import load_word_embedding
from bean.file_path import *
from data_batches.batches import dataset_iterator


class cnn_model(object):
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
        self.x_train, self.y_train, self.word_index, self.x_dev, self.y_dev, self.max_len = \
            prepocess(mr_pos_path, mr_neg_path)

        self.x_inputs = tf.placeholder(tf.int32, [None, self.max_len], name='x_inputs')
        self.y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_inputs')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.all_test_acc = []

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
            embedded_chars = tf.nn.embedding_lookup(glove_w2v, self.x_inputs)
            sen_inputs_glove = tf.expand_dims(embedded_chars, -1)
        return sen_inputs_glove

    def cnn_pool(self):
        sen_inputs_glove = self.embedding_layer()

        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
            conv = tf.nn.conv2d(
                sen_inputs_glove,
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
        return pooled_outputs

    # 全连接
    def fully_connect(self, num_filters_total):
        pooled_outputs = self.cnn_pool()
        # combine all the pooled fratures
        h_pool = tf.concat(pooled_outputs, axis=3)  # [b_s, 1, 1, 384]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # [64, 384]
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return h_drop

    # softmax
    def softmax_output(self):
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_drop = self.fully_connect(num_filters_total)

        W = tf.get_variable(
            name='W',
            shape=[num_filters_total, self.num_classes],
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

