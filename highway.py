def highway_layer(input_data, dim, init, name='', reuse=None):
    """ Creates a highway layer
    """
    trans = linear(input_data, dim, init, name='trans_{}'.format(name),
                   reuse=reuse)
    trans = tf.nn.relu(trans)
    gate = linear(input_data, dim, init, name='gate_{}'.format(name),
                  reuse=reuse)
    gate = tf.nn.sigmoid(gate)
    if dim != input_data.get_shape()[-1]:
        input_data = linear(input_data, dim, init, name='trans2_{}'.format(name),
                            reuse=reuse)
    output = gate * trans + (1 - gate) * input_data
    return output


def linear(input_data, dim, initializer, name='', reuse=None):
    """ Default linear layer
    """
    input_shape = input_data.get_shape().as_list()[1]
    with tf.variable_scope('linear', reuse=reuse) as scope:
        _weights = tf.get_variable(
            "W_{}".format(name),
            shape=[input_shape, dim],
            initializer=initializer)
        _bias = tf.get_variable('bias_{}'.format(name),
                                shape=[dim],
                                initializer=tf.constant_initializer([0.1]))
    output_data = tf.nn.xw_plus_b(input_data, _weights, _bias)
    return output_data
