    import tensorflow as tf
    @staticmethod
    def Cosine_Similarity(x, y):
        return tf.reduce_sum(x * y, axis=-1) / (
            tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1)))
