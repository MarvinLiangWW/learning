import tensorflow as tf


class Optimize(object):
    def __init__(self, args):
        self.args = args

    def hinge_loss(self, ):
        # Define loss and optimizer
        
        # self.output_pos and self.output_neg 
        
        with tf.name_scope("train"):
            with tf.name_scope("cost_function"):
                # hinge loss
                self.hinge_loss = tf.maximum(0.0, (
                    self.args.margin - self.output_pos + self.output_neg))
                self.cost = tf.reduce_sum(self.hinge_loss)
                with tf.name_scope('regularization'):
                    if self.args.l2_reg > 0:
                        vars = tf.trainable_variables()
                        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
                        lossL2 *= self.args.l2_reg
                        self.cost += lossL2
                tf.summary.scalar("cost_function", self.cost)
            global_step = tf.Variable(0, trainable=False)

            if self.args.decay_lr > 0 and self.args.decay_epoch > 0:
                decay_epoch = self.args.decay_epoch
                lr = tf.train.exponential_decay(self.args.learn_rate, global_step,
                                                decay_epoch * self.args.batch_size,
                                                self.args.decay_lr, staircase=True)
            else:
                lr = self.args.learn_rate

            with tf.name_scope('optimizer'):
                if self.args.opt == 'SGD':
                    self.opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
                elif self.args.opt == 'Adam':
                    self.opt = tf.train.AdamOptimizer(learning_rate=lr)
                elif self.args.opt == 'Moment':
                    self.opt = tf.train.MomentumOptimizer(lr, 0.9)
