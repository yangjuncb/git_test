import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sentence1_length, sentence2_length, num_classes,
                 vocab_size,  embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_s1 = tf.placeholder(tf.int32, [None, sentence1_length], name="input_sentence1")
        self.input_s2 = tf.placeholder(tf.int32, [None, sentence2_length], name="input_sentence2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        #used to extract features from input sentence
        def sentence_model(input_x, sentence_length, name_prefix='s1'):
            #Embedding layer
            with tf.device('/cpu:0'), tf.name_scope(name_prefix+"embedding"):
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.l2_loss += tf.nn.l2_loss(W)
                embedded_chars = tf.nn.embedding_lookup(W, input_x)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope(name_prefix+"conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    self.l2_loss += tf.nn.l2_loss(W)
                    self.l2_loss += tf.nn.l2_loss(b)
                    conv = tf.nn.conv2d(
                            embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sentence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            return h_pool_flat, num_filters_total

        #extract sentence1 features, such as query
        self.h_pool_flat_s1, self.len_s1 = sentence_model(self.input_s1,
                                                          self.sentence1_length,
                                                          name='s1')

        #extract sentence2 features, such as title or document
        self.h_pool_flat_s2, self.len_s2 = sentence_model(self.input_s2,
                                                          self.sentence2_length,
                                                          name='s2')
        #similarity matching
        with tf.name_scope('similarity_matching'):
            W = tf.get_variable(
                'W',
                shape=[self.len_s1, self.len_s2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', tf.constant(.1))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.X_sim = tf.add(tf.matmul(tf.matmul(self.h_pool_flat_s1, W),
                                          self.h_pool_flat_s2,
                                          transpose_b = True),
                                b)

        #join layer
        with tf.name_scope('concat'):
            self.join = tf.concat(1,
                                  [self.h_pool_flat_s1, self.X_sim, self.h_pool_flat_s2])

        #hidden layer
        with tf.name_scope('hidden'):
             self.hidden_size = self.len_s1 + self.len_s2 + 1
             W = tf.get_variable(
                 'W',
                 shape = [self.hidden_size, self.hidden_size],
                 initializer = tf.contrib.layers.xavier_initializer())
             b = tf.get_variable('b', tf.constant(.1, shape=[self.hidden_size]))
             self.l2_loss += tf.nn.l2_loss(W)
             self.l2_loss += tf.nn.l2_loss(b)
             self.hidden = tf.nn.xw_plus_b(self.join, W, b, 'hidden')

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
