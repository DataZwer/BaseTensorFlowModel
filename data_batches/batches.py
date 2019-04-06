import math
import tensorflow as tf


def dataset_iterator(x_input, y_input, data_len):
    batch_size = 64
    train_nums = data_len
    iterations = math.ceil(train_nums / batch_size)  # 总共可以划分出来的batch数量

    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))
    dataset = dataset.batch(batch_size).repeat()

    # 使用生成器make_one_shot_iterator和get_next取数据
    iterator = dataset.make_one_shot_iterator()
    next_iterator = iterator.get_next()
    return iterations, next_iterator
