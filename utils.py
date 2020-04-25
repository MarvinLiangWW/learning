class UTILS:
    nb_topk = 50
    mid_pop = 5
    show_vs = {}

    @staticmethod
    def get_session():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1,
            visible_device_list=args.gpu,
            allow_growth=True,
        )
        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        return session

    @staticmethod
    def minimizer(loss):
        loss += tf.losses.get_regularization_loss()
        opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        minimizer_op = opt.minimize(loss)
        return minimizer_op

    @staticmethod
    def l2_loss(name):
        alpha = args.get(f'l2_{name}', 0)
        if alpha < 1e-7:
            return None
        return lambda x: alpha * tf.nn.l2_loss(x)

    @staticmethod
    def mask_logits(logits, mask):
        mask = tf.cast(mask, tf.float32)
        logits = logits * mask - (1 - mask) * 1e12
        return logits

    @staticmethod
    def LSTM(seq, seq_length=None, name='text'):
        k = seq.get_shape().as_list()[-1]
        with tf.variable_scope(f'LSTM_{name}', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.LSTMCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.dynamic_rnn(cell, seq, sequence_length=seq_length, dtype=tf.float32)
            return state.h

    @staticmethod
    def GRU(seq, seq_length=None, mask=None, name='text'):
        k = seq.get_shape().as_list()[-1]
        if seq_length is None and mask is not None:
            seq_length = tf.reduce_sum(tf.cast(mask, tf.int32), -1)
        with tf.variable_scope(f'GRU_{name}', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.GRUCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.dynamic_rnn(cell, seq, sequence_length=seq_length, dtype=tf.float32)
            return outputs, state

    @staticmethod
    def BiRNN(seq, seq_length=None, name='text'):
        k = args.dim_k
        with tf.variable_scope(f'BiGRU_{name}', reuse=tf.AUTO_REUSE):
            cell_f = tf.nn.rnn_cell.GRUCell(k)
            cell_b = tf.nn.rnn_cell.GRUCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, seq, sequence_length=seq_length, dtype=tf.float32)
        return outputs

    @staticmethod
    def Mean(seq, seq_length=None, mask=None, name=None):
        # seq: (None, L, k), seq_length: (None, ), mask: (None, L)
        # ret: (None, k)
        if seq_length is None and mask is None:
            with tf.variable_scope('Mean'):
                return tf.reduce_sum(seq, -2)

        with tf.variable_scope('MaskMean'):
            if mask is None:
                mask = tf.sequence_mask(seq_length, maxlen=tf.shape(seq)[1], dtype=tf.float32)
            mask = tf.expand_dims(mask, -1)  # (None, L, 1)
            seq = seq * mask
            seq = tf.reduce_sum(seq, -2)  # (None, k)
            seq = seq / (tf.reduce_sum(mask, -2) + eps)
        return seq

    @staticmethod
    def MLP(x, fc, activation, name):
        with tf.variable_scope(f'MLP_{name}'):
            for i in range(len(fc)):
                x = tf.layers.dense(x, fc[i], activation=activation, name=f'dense_{i}')
        return x

    @staticmethod
    def gate(a, b, name):
        with tf.variable_scope(name):
            alpha = tf.layers.dense(tf.concat([a, b], -1), 1, activation=tf.nn.sigmoid, name='gateW')
            ret = alpha * a + (1 - alpha) * b
        return ret

    @staticmethod
    def dot(a, b):
        return tf.reduce_sum(a * b, -1)

    @staticmethod
    def Embedding(node, n, name='node'):
        # node: [BS]
        with tf.variable_scope(f'Emb_{name}'):
            k = args.dim_k
            emb_w = tf.get_variable(
                name='emb_w',
                shape=[n, k],
                # initializer=tf.random_normal_initializer(0.0, 0.05),
            )
            t = tf.gather(emb_w, node)
            mask = tf.not_equal(node, 0)
            mask = tf.cast(mask, tf.float32)
            t = t * tf.cast(tf.expand_dims(mask, -1), tf.float32)
        return t, mask, emb_w

    @staticmethod
    def dense(x, k, use_bias=True, name='dense', **kwargs):
        init = tf.glorot_normal_initializer()
        ret = tf.layers.dense(x, k, use_bias=use_bias, name=name, kernel_initializer=init, **kwargs)
        return ret

    @staticmethod
    def save_npy(ar, fn):
        fn = f'{utils.data_dir}/{args.ds}/{fn}'
        np.save(fn, ar)
    @staticmethod
    def load_npy(fn):
        fn = f'{utils.data_dir}/{args.ds}/{fn}'
        return np.load(fn)

    @staticmethod
    def last_e(seq, mask=None):
        if mask is None:
            # [BS, L]
            mask = tf.not_equal(seq, 0)
        # [BS, 1]
        length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1, keepdims=True)
        # [BS, 1]
        e = tf.batch_gather(seq, length - 1)
        return e
