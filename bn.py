# Batch normalization implementation
# from https://github.com/tensorflow/tensorflow/issues/1122
def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                    lambda: batch_norm(inputT, is_training=True,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                    lambda: batch_norm(inputT, is_training=False,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                    scope=scope, reuse = True))

# Create model of MLP with batch-normalization layer
def MLPwithBN(x, weights, biases, is_training=True):
    with tf.name_scope('MLPwithBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = batch_norm_layer(layer_1,is_training=is_training, scope='layer_1_bn')
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = batch_norm_layer(layer_2, is_training=is_training, scope='layer_2_bn')
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Create model of MLP without batch-normalization layer
def MLPwoBN(x, weights, biases):
    with tf.name_scope('MLPwoBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# main function
def main():
    # Parse argument
    parser = build_parser()
    options = parser.parse_args()
    weights = weight_initializer[options.weight_initializer]
    biases = bias_initializer[options.bias_initializer]
    batch_normalization = options.batch_normalization

    # Import data
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) #answer

    # Predict
    if batch_normalization=='True':
        y = MLPwithBN(x,weights,biases,is_training)
    else:
        y = MLPwoBN(x, weights, biases)

    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

    # Define optimizer
    with tf.name_scope("ADAM"):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # moving_mean and moving_variance need to be updated
    if batch_normalization == "True":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            train_ops = [train_step] + update_ops
            train_op_final = tf.group(*train_ops)
        else:
            train_op_final = train_step

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor loss tensor
    tf.scalar_summary('loss', loss)

    # Create a summary to monitor accuracy tensor
    tf.scalar_summary('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.merge_all_summaries()

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(mnist.train.num_examples / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.train.SummaryWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Loop for epoch
    for epoch in range(training_epochs):

        # Loop over all batches
        for i in range(total_batch):

            batch = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_op_final, accuracy, merged_summary_op] , feed_dict={x: batch[0], y_: batch[1], is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # Calculate accuracy for all mnist test images
    print("test accuracy for the latest result: %g" % accuracy.eval(
    feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training: False}))

if __name__ == '__main__':
    main()
