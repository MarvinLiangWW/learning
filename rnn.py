class rnn(object):
    @staticmethod
    def _len(sequence):
        return tf.reduce_sum(tf.sign(sequence), axis=-1)

    @staticmethod
    def _weight_and_bias(in_size, out_size, name):
        weight = tf.get_variable('weight_{}'.format(name), shape=[in_size, out_size],
                                 initializer=tf.random_normal_initializer(), )
        bias = tf.get_variable('bias_{}'.format(name), shape=[out_size, ], initializer=tf.constant_initializer([0.1]))
        return weight, bias

    def last_relevant(self, output, length):
        '''
        :param output: lstm output
        :param length: relevant last index
        :return: (batch_size, rnn_dim)
        '''
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = tf.shape(output)[2]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

