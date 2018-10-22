import tensorflow as tf

import numpy as np
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_tf import initialize_uninitialized_global_variables
from discretization_utils import discretize_uniform
import sys
import time


class BaseModel:
    def __init__(self, x, y, sess, var_scope='BaseModel', path=None):
        """
        :param x: tf.placeholder -- input to the model
        :param var_scope: string -- change this parameter if you want to create multiple instances from this model
        """
        self.var_scope = var_scope
        self.path = path
        self.x = x
        self.y = y
        self.sess = sess
        self.optimizer_dic = {}
        self.build_graph(x)
        self.labels = tf.argmax(self.probs, axis=1)

    def build_graph(self, x, reuse=False, self_update=True):
        """

        :type self_update: bool
        """
        print self.var_scope
        print 'nothing more...'

    def reuse_graph(self, x, self_update=False):
        """

        :rtype: (probs, logits)
        :param x: tensorflow placeholder
        :param self_update: set it to True if you want to get intermediate layer outputs from input x
        :return: probs and logits of model when x is fed into it
        """
        return self.build_graph(x, reuse=True, self_update=self_update)

    # def get_labels(self):
    #     return tf.argmax(self.probs, axis=1)

    def load_model(self, path):
        g = tf.get_default_graph()
        var_list = []
        for i in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.var_scope):
            var_list.append(i)
        saver = tf.train.Saver(var_list)
        saver.restore(self.sess, path)

    def train_model(self, x_train_, y_train_, lr=0.001, nb_epochs=10, other_params={}):
        batch_size = other_params.get('batch_size', 128)
        optimizer_name = other_params.get('optimizer_name', 'Adam')
        if lr in self.optimizer_dic:
            optimizer, loss, optim_step = self.optimizer_dic[lr]
        else:
            if optimizer_name == 'Adam':
                optimizer = tf.train.AdamOptimizer(lr, name='vanilla_' + self.var_scope + '_' + str(lr))
            else:
                optimizer = tf.train.GradientDescentOptimizer(lr, name='vanilla_' + self.var_scope + '_' + str(lr))
            probs, logits = self.reuse_graph(self.x)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
            loss = tf.reduce_mean(loss)
            optim_step = optimizer.minimize(loss)
            initialize_uninitialized_global_variables(self.sess)
            self.optimizer_dic[lr] = (optimizer, loss, optim_step)
        # Shuffling dataset
        assert len(x_train_) == len(y_train_)
        p = np.random.permutation(len(x_train_))
        x_train, y_train = x_train_[p], y_train_[p]
        # For considering last portion of data
        total_batch = len(x_train) / batch_size
        if len(x_train) % batch_size != 0:
            total_batch += 1
        big_loss = 0
        big_counter = 0
        for ep in range(nb_epochs):
            # print 'epoch: ', ep
            for i in range(total_batch):
                x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                _, loss_np = self.sess.run([optim_step, loss], {self.x: x_batch, self.y: y_batch})
                big_loss += loss_np
                big_counter += 1
                if i % 50 == 0:
                    # print i, loss_np
                    mean_loss = big_loss / big_counter
                    print '\b' * 1000,
                    print '\r' + 'epoch: ' + str(ep + 1) + '/' + str(nb_epochs) + \
                          ' iter: ' + str(i) + '/' + str(total_batch) + ' loss: ' + \
                          "{0:0.4f}".format(mean_loss),
                    sys.stdout.flush()
        print ''

    def test_model(self, x_test, y_test, batch_size=128):
        total_batch = len(x_test) / batch_size
        if len(x_test) % batch_size != 0:
            total_batch += 1
        total_correct = 0
        for i in range(total_batch):
            x_batch = x_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size].argmax(axis=1)
            temp_labels = self.sess.run(self.labels, {self.x: x_batch})
            total_correct += np.sum(temp_labels == y_batch)
        print 'model accuracy is:', total_correct / (len(x_test) + .0)

    def get_noisy_data_and_logits(self, x_train, nb_replica=7, start_range=0, end_range=100, step=1, batch_size=128):
        x_train_noisy = np.copy(x_train)
        for i in range(nb_replica - 1):
            x_train_noisy = np.concatenate((x_train_noisy, x_train), axis=0)
        print 'x_train_noisy.shape', x_train_noisy.shape
        ranges = (end_range - start_range) / step
        total_batch = len(x_train_noisy) / batch_size
        if len(x_train_noisy) % batch_size != 0:
            total_batch += 1
        logits_noisy = np.zeros((len(x_train_noisy), 10))
        counter = 0
        total_iters = total_batch
        st = time.time()
        for i in range(total_batch):
            counter += 1
            x_batch = x_train_noisy[i * batch_size:(i + 1) * batch_size]
            noise_level = np.random.randint(0, ranges + 1) * step + start_range
            rand_noise = (np.random.random(x_batch.shape) - 0.5) * 0.02 * noise_level
            x_batch = np.clip(x_batch + rand_noise, 0., 1.)
            x_train_noisy[i * batch_size:(i + 1) * batch_size] = x_batch
            y_batch = self.sess.run(self.logits, {self.x: x_batch})
            logits_noisy[i * batch_size:(i + 1) * batch_size] = y_batch
            if i % 10 == 9:
                remained_iters = total_iters - counter
                passed_time = time.time() - st
                ETA = int(passed_time * remained_iters / counter)
                ETA_min, ETA_sec = ETA / 60, ETA % 60
                print '\b' * 1000,
                print '\r' + ' iter: ' + str(i + 1) + '/' + str(total_batch) + \
                      ' ETA: ' + str(ETA_min) + ':' +"{0:02d}".format(ETA_sec),
                sys.stdout.flush()
        print ''
        return  x_train_noisy,logits_noisy
    def SST_v2(self, x_train_, y_train_, lr=0.001, nb_epochs=10, batch_size=128):
        """

        :param x_train_:
        :param target_model:
        :param lr:
        :param nb_epochs:
        :param start_range: an int number from 0 to 100. Shows the minimum level of random noise
        :param end_range: an int number from 0 to 100. Shows the maximum level of random noise
        :param step: an int number from 1 to 100.
        :return:
        """
        if lr in self.optimizer_dic:
            optimizer, loss, optim_step = self.optimizer_dic[lr]
        else:
            optimizer = tf.train.AdamOptimizer(lr, name='vanilla_' + self.var_scope + '_' + str(lr))
            probs, logits = self.reuse_graph(self.x)
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            # loss = tf.reduce_mean(loss)
            loss = tf.reduce_mean(tf.square(self.logits - self.y))
            optim_step = optimizer.minimize(loss)
            initialize_uninitialized_global_variables(self.sess)
            self.optimizer_dic[lr] = (optimizer, loss, optim_step)
        # Shuffling dataset
        p = np.random.permutation(len(x_train_))
        x_train,y_train = x_train_[p],y_train_[p]
        # For considering last portion of data
        total_batch = len(x_train) / batch_size
        if len(x_train) % batch_size != 0:
            total_batch += 1
        big_loss = 0
        big_counter = 0
        counter = 0
        total_iters = nb_epochs * total_batch
        st = time.time()
        for ep in range(nb_epochs):
            # print 'epoch: ', ep
            for i in range(total_batch):
                counter += 1
                x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                _, loss_np = self.sess.run([optim_step, loss], {self.x: x_batch, self.y: y_batch})
                big_loss += loss_np
                big_counter += 1
                if i % 50 == 49:
                    remained_iters = total_iters - counter
                    passed_time = time.time() - st
                    ETA = int(passed_time * remained_iters / counter)
                    ETA_min, ETA_sec = ETA / 60, ETA % 60
                    mean_loss = big_loss / big_counter
                    print '\b' * 1000,
                    print '\r' + 'epoch: ' + str(ep + 1) + '/' + str(nb_epochs) + \
                          ' iter: ' + str(i + 1) + '/' + str(total_batch) + \
                          ' ETA: ' + str(ETA_min) + ':' + "{0:02d}".format(ETA_sec) + \
                          ' loss: ' + "{0:0.4f}".format(mean_loss),
                    sys.stdout.flush()
        print ''

    def SST(self, x_train_, target_model, lr=0.001, nb_epochs=10,
            start_range=0, end_range=100, step=1, batch_size=128):
        """

        :param x_train_:
        :param target_model:
        :param lr:
        :param nb_epochs:
        :param start_range: an int number from 0 to 100. Shows the minimum level of random noise
        :param end_range: an int number from 0 to 100. Shows the maximum level of random noise
        :param step: an int number from 1 to 100.
        :return:
        """
        if lr in self.optimizer_dic:
            optimizer, loss, optim_step = self.optimizer_dic[lr]
        else:
            optimizer = tf.train.AdamOptimizer(lr, name='vanilla_' + self.var_scope + '_' + str(lr))
            probs, logits = self.reuse_graph(self.x)
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
            # loss = tf.reduce_mean(loss)
            loss = tf.reduce_mean(tf.square(self.logits - self.y))
            optim_step = optimizer.minimize(loss)
            initialize_uninitialized_global_variables(self.sess)
            self.optimizer_dic[lr] = (optimizer, loss, optim_step)
        # Shuffling dataset
        p = np.random.permutation(len(x_train_))
        x_train = x_train_[p]
        # For considering last portion of data
        total_batch = len(x_train) / batch_size
        if len(x_train) % batch_size != 0:
            total_batch += 1
        ranges = (end_range - start_range) / step
        big_loss = 0
        big_counter = 0
        counter = 0
        total_iters = nb_epochs * total_batch
        st = time.time()
        for ep in range(nb_epochs):
            # print 'epoch: ', ep
            for i in range(total_batch):
                counter += 1
                x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                noise_level = np.random.randint(0, ranges + 1) * step + start_range
                rand_noise = (np.random.random(x_batch.shape) - 0.5) * 0.02 * noise_level
                x_batch = np.clip(x_batch + rand_noise, 0., 1.)
                y_batch = target_model.sess.run(target_model.logits, {target_model.x: x_batch})
                _, loss_np = self.sess.run([optim_step, loss], {self.x: x_batch, self.y: y_batch})
                big_loss += loss_np
                big_counter += 1
                if i % 50 == 49:
                    remained_iters = total_iters - counter
                    passed_time = time.time() - st
                    ETA = int(passed_time * remained_iters / counter)
                    ETA_min, ETA_sec = ETA / 60, ETA % 60
                    mean_loss = big_loss / big_counter
                    print '\b' * 1000,
                    print '\r' + 'epoch: ' + str(ep + 1) + '/' + str(nb_epochs) + \
                          ' iter: ' + str(i + 1) + '/' + str(total_batch) + \
                          ' ETA: ' + str(ETA_min) + ':' + "{0:02d}".format(ETA_sec)+ \
                          ' loss: ' + "{0:0.4f}".format(mean_loss),
                    sys.stdout.flush()
        print ''

    def callable_model(self, type='probs'):  # type : 'probs' or 'logits'
        """
        :param type: 'probs' or 'logits'
        :return: Cleverhans Model
        """

        def wrapper(x):
            probs, logits = self.reuse_graph(x)
            if type == 'probs':
                return probs
            else:
                return logits

        model = CallableModelWrapper(wrapper, type)
        return model


class SimpleCNN(BaseModel, object):
    def __init__(self, x, y, sess, var_scope='SimpleCNN', path=None):
        super(SimpleCNN, self).__init__(x, y, sess, var_scope, path)

    def build_graph(self, x, reuse=False, self_update=True):
        # print self.var_scope
        with tf.variable_scope(self.var_scope, reuse=reuse):
            inp = x
            conv1 = tf.layers.conv2d(inp, 64, 3, 1, activation=tf.nn.relu, name='conv1')
            maxpool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool1')
            conv2 = tf.layers.conv2d(maxpool1, 64, 3, 1, activation=tf.nn.relu, name='conv2')
            flatten = tf.contrib.layers.flatten(conv2)
            dense1 = tf.layers.dense(flatten, 2048, tf.nn.relu, name='dense1')
            dense2 = tf.layers.dense(dense1, 10, name='dense2')
            logits = dense2
            probs = tf.nn.softmax(dense2)
        if self_update:
            self.conv1 = conv1
            self.maxpool1 = maxpool1
            self.conv2 = conv2
            self.flatten = flatten
            self.dense1 = dense1
            self.dense2 = dense2
            self.logits = logits
            self.probs = probs
        if not reuse and self.path:
            self.load_model(self.path)
        return probs, logits


class SimpleCNN2(BaseModel, object):
    def __init__(self, x, y, sess, var_scope='SimpleCNN2', path=None):
        super(SimpleCNN2, self).__init__(x, y, sess, var_scope, path)

    def build_graph(self, x, reuse=False, self_update=True):
        # print self.var_scope
        with tf.variable_scope(self.var_scope, reuse=reuse):
            inp = x
            conv1 = tf.layers.conv2d(inp, 20, 7, 1, activation=tf.nn.relu, name='conv1')
            maxpool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool1')
            conv2 = tf.layers.conv2d(maxpool1, 40, 3, 1, activation=tf.nn.relu, name='conv2')
            flatten = tf.contrib.layers.flatten(conv2)
            dense1 = tf.layers.dense(flatten, 1500, tf.nn.relu, name='dense1')
            dense2 = tf.layers.dense(dense1, 500, tf.nn.relu, name='dense2')
            dense3 = tf.layers.dense(dense2, 10, name='dense3')
            logits = dense3
            probs = tf.nn.softmax(dense3)
        if self_update:
            self.conv1 = conv1
            self.maxpool1 = maxpool1
            self.conv2 = conv2
            self.flatten = flatten
            self.dense1 = dense1
            self.dense2 = dense2
            self.logits = logits
            self.probs = probs
        if not reuse and self.path != None:
            self.load_model(self.path)
        return probs, logits


class SimpleCNN3(BaseModel, object):
    def __init__(self, x, y, sess, var_scope='SimpleCNN3', path=None):
        super(SimpleCNN3, self).__init__(x, y, sess, var_scope, path)

    def build_graph(self, x, reuse=False, self_update=True):
        # print self.var_scope
        with tf.variable_scope(self.var_scope, reuse=reuse):
            inp = x
            conv1 = tf.layers.conv2d(inp, 64, 3, 1, padding='same', activation=tf.nn.relu, name='conv1')
            conv2 = tf.layers.conv2d(conv1, 64, 3, 1, padding='same', activation=tf.nn.relu, name='conv2')
            conv3 = tf.layers.conv2d(conv2, 64, 3, 1, padding='valid', activation=tf.nn.relu, name='conv3')
            maxpool1 = tf.layers.max_pooling2d(conv3, 2, 2, name='maxpool1')
            conv4 = tf.layers.conv2d(maxpool1, 64, 3, 1, activation=tf.nn.relu, name='conv4')
            flatten = tf.contrib.layers.flatten(conv4)
            dense1 = tf.layers.dense(flatten, 2048, tf.nn.relu, name='dense1')
            dense2 = tf.layers.dense(dense1, 10, name='dense2')
            logits = dense2
            probs = tf.nn.softmax(dense2)
        if self_update:
            self.conv1 = conv1
            self.conv2 = conv2
            self.conv3 = conv3
            self.maxpool1 = maxpool1
            self.conv4 = conv4
            self.flatten = flatten
            self.dense1 = dense1
            self.dense2 = dense2
            self.logits = logits
            self.probs = probs
        if not reuse and self.path:
            self.load_model(self.path)
        return probs, logits


class RFNModel(BaseModel, object):
    def __init__(self, x, y, sess, var_scope='RFNModel', path=None):
        self.training = True
        super(RFNModel, self).__init__(x, y, sess, var_scope, path)

    def build_graph(self, x, reuse=False, self_update=True):
        with tf.variable_scope(self.var_scope, reuse=reuse):
            rand_vec = tf.random_uniform([100, 28 * 28 * 1], 0., 1.)
            flatten = tf.contrib.layers.flatten(x)
            if self.training:
                norm_dist = tf.random_normal([100, 1], 0.5, 0.05)
                gg = (rand_vec > norm_dist)
                gg = tf.multiply(tf.cast(gg, tf.float32), 1)
                masked = flatten * gg
            else:
                gg = (rand_vec > 0.5)
                gg = tf.multiply(tf.cast(gg, tf.float32), 1)
                masked = flatten * gg

            dense1 = tf.layers.dense(masked, 784, tf.nn.relu, name='dense1')
            dropout1 = tf.layers.dropout(dense1, rate=0.25, training=self.training)
            dense2 = tf.layers.dense(dropout1, 784, tf.nn.relu, name='dense2')
            dropout2 = tf.layers.dropout(dense2, rate=0.25, training=self.training)
            dense3 = tf.layers.dense(dropout2, 784, tf.nn.relu, name='dense3')
            dropout3 = tf.layers.dropout(dense3, rate=0.25, training=self.training)
            dense4 = tf.layers.dense(dropout3, 784, tf.nn.relu, name='dense4')
            dropout4 = tf.layers.dropout(dense4, rate=0.25, training=self.training)
            dense5 = tf.layers.dense(dropout4, 10, name='dense5')
            logits = dense5
            probs = tf.nn.softmax(logits)
        if self_update:
            self.flatten = flatten
            self.masked = masked
            self.dense1 = dense1
            self.dropout1 = dropout1
            self.dense2 = dense2
            self.dropout2 = dropout2
            self.dense3 = dense3
            self.dropout3 = dropout3
            self.dense4 = dense4
            self.dropout4 = dropout4
            self.dense5 = dense5
            self.logits = logits
            self.probs = probs
            self.labels = tf.argmax(self.probs, axis=1)

        if not reuse and self.path:
            self.load_model(self.path)
        return probs, logits


class ThermModel(BaseModel, object):
    def __init__(self, x, y, sess, therm_model):
        self.therm_model = therm_model
        super(ThermModel, self).__init__(x, y, sess)

    def build_graph(self, x, reuse=False, self_update=True):
        encode = discretize_uniform(x, levels=16, thermometer=True)
        logits = self.therm_model(encode)
        probs = tf.nn.softmax(logits)
        if self_update:
            self.logits = logits
            self.probs = probs
            self.labels = tf.argmax(self.probs, axis=1)
        return probs, logits

    def load_model(self, path):
        print 'This method is not supported for Thermometer Encoding class!'


class SimpleCNN4(BaseModel, object):
    def __init__(self, x, y, sess, var_scope='SimpleCNN4', path=None):
        super(SimpleCNN4, self).__init__(x, y, sess, var_scope, path)

    def build_graph(self, x, reuse=False, self_update=True):
        # print self.var_scope
        with tf.variable_scope(self.var_scope, reuse=reuse):
            inp = x
            conv1 = tf.layers.conv2d(inp, 64, 3, 1, padding='same', activation=tf.nn.relu, name='conv1')
            conv2 = tf.layers.conv2d(conv1, 64, 3, 1, padding='same', activation=tf.nn.relu, name='conv2')
            maxpool1 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool1')
            conv3 = tf.layers.conv2d(maxpool1, 128, 3, 1, activation=tf.nn.relu, name='conv3')
            conv4 = tf.layers.conv2d(conv3, 64, 3, 1, activation=tf.nn.relu, name='conv4')
            maxpool2 = tf.layers.max_pooling2d(conv4, 2, 2, name='maxpool2')
            flatten = tf.contrib.layers.flatten(maxpool2)
            dense1 = tf.layers.dense(flatten, 4096, tf.nn.relu, name='dense1')
            dense2 = tf.layers.dense(dense1, 1024, tf.nn.relu, name='dense2')
            dense3 = tf.layers.dense(dense2, 10, name='dense3')
            logits = dense3
            probs = tf.nn.softmax(dense3)
        if self_update:
            self.conv1 = conv1
            self.conv2 = conv2
            self.maxpool1 = maxpool1
            self.conv3 = conv3
            self.conv4 = conv4
            self.maxpool2 = maxpool2
            self.flatten = flatten
            self.dense1 = dense1
            self.dense2 = dense2
            self.dense3 = dense3
            self.logits = logits
            self.probs = probs
        if not reuse and self.path:
            self.load_model(self.path)
        return probs, logits
