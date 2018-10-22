import tensorflow as tf
import numpy as np
import models as Mo
from cleverhans.utils_tf import initialize_uninitialized_global_variables
import time

class Attack(object):

    def __init__(self, x, y, sub_models, target_model, sess, batch_size=10, lr=1 / 255., y_target=None,eps=0):
        """

        :param x:
        :param sub_models:
        :param target_model:
        :param lr:
        :param target: One of "targeted","non-targeted","
        """
        self.x = x
        self.y = y
        self.eps = eps
        self.target_model = target_model
        self.sub_models = sub_models
        self.sess = sess
        self.batch_size = batch_size
        pert_shape = [self.batch_size]
        pert_shape.extend(x.shape.as_list()[1:])
        self.pert_var = tf.Variable(np.zeros(pert_shape, dtype=np.float32), name='perturbation')
        if eps==0:
            adv_img_unbounded = self.pert_var + x
            self.adv_img = tf.clip_by_value(adv_img_unbounded, 0.0, 1.0)
        else:
            adv_img_bounded = tf.clip_by_value(self.pert_var,-eps,eps) + x
            self.adv_img = tf.clip_by_value(adv_img_bounded, 0.0, 1.0)
        target_probs, _ = self.target_model.reuse_graph(self.adv_img)
        self.target_model_out = tf.argmax(target_probs, axis=1)

        self.all_adv_logits = []
        for sub_model in sub_models:
            _, temp_adv_logits = sub_model.reuse_graph(self.adv_img)
            self.all_adv_logits.append(temp_adv_logits)
        f_loss = 0
        self.kappa = tf.placeholder(tf.float32, [])
        self.c = tf.placeholder(tf.float32, [self.batch_size])
        self.y_target = y_target
        if y_target is not None:
            for logits in self.all_adv_logits:
                correct_logits = tf.reduce_sum(y * logits, axis=1)
                target_logits = tf.reduce_sum(y_target * logits, axis=1)
                f_loss += tf.maximum(correct_logits - target_logits + self.kappa, 0)
        else:
            for logits in self.all_adv_logits:
                correct_logits = tf.reduce_sum(y * logits, axis=1)
                wrong_logits = tf.reduce_max((1 - y) * logits - 1e4 * y, axis=1)
                f_loss += tf.maximum(correct_logits - wrong_logits + self.kappa, 0)
        if eps==0:
            p_loss = self.c * tf.reduce_sum(self.pert_var ** 2, axis=[1, 2, 3]) / 2.
        else:
            p_loss = self.c * tf.reduce_sum(tf.abs(self.pert_var), axis=[1, 2, 3]) / 2.
        self.loss = f_loss + p_loss

        self.optimizer = tf.train.AdamOptimizer(lr, name='cw_adam')
        self.optim_step = self.optimizer.minimize(self.loss, var_list=[self.pert_var])
        initialize_uninitialized_global_variables(self.sess)

    def find_adv(self, img, true_label, target_label=None,other_params={}):
        total_run = other_params.get('total_run',3)
        total_iters = other_params.get('total_iters',300)
        adv_condition = other_params.get('adv_condition',None)
        kappa_vals = [1, 5, 10, 25, 40, 80]
        if 'kappa_vals' in other_params:
            kappa_vals = other_params['kappa_vals']
        for i in range(len(kappa_vals),total_run):
            kappa_vals.append(200)

        im_sh = img.shape
        best_adv_big = np.ones([total_run, im_sh[0], im_sh[1], im_sh[2], im_sh[3]]) * 10
        self.sess.run(self.pert_var.initializer)
        self.pert_var.load((np.random.random((self.batch_size, im_sh[1], im_sh[2], im_sh[3])) - 0.5) * 0.1)
        total_adv_found = np.zeros((self.batch_size))
        total_run_iters = total_run*total_iters
        counter = 0
        st = time.time()
        for run in range(total_run):
            best_adv = np.ones(img.shape) * 10

            c_np = np.zeros((self.batch_size)) + 6.
            adv_found = [False] * self.batch_size
            kappa_np = kappa_vals[run]
            for iter in range(total_iters):
                counter+=1
                feed_dict = {self.x: img, self.y: true_label, self.kappa: kappa_np, self.c: c_np}
                if self.y_target:
                    feed_dict[self.y_target] = target_label
                _, loss_np = self.sess.run([self.optim_step, self.loss], feed_dict)
                if iter % 50 == 49:
                    remained_iters = total_run_iters - counter
                    passed_time = time.time() - st
                    ETA = int(passed_time * remained_iters / counter)
                    ETA_min, ETA_sec = ETA / 60, ETA % 60
                    total_adv_found = total_adv_found + np.array(adv_found)
                    print '\b' * 1000,
                    print '\r' + 'run: ' + str(run+1) + '/' + str(total_run) + \
                          ' iter: ' + str(iter+1) + '/' + str(total_iters) + ' ETA: ' +\
                          str(ETA_min) + ':' +"{0:02d}".format(ETA_sec) +\
                          ' success rate: ' + \
                          "{0:0.4f}".format(np.sum(np.array(total_adv_found)!=0)/(self.batch_size+0.)),
                adv_img_np = self.sess.run(self.adv_img, {self.x: img})
                if isinstance(adv_condition,type(None)):
                    target_model_out_np = self.sess.run(self.target_model.labels, {self.target_model.x: adv_img_np})
                    temp_cond = target_model_out_np!=true_label.argmax(axis=1)
                else:
                    temp_cond = adv_condition(adv_img_np,true_label)
                for im in range(len(img)):
                    if temp_cond[im] and \
                            np.linalg.norm(adv_img_np[im] - img[im]) < np.linalg.norm(best_adv[im] - img[im]):
                        best_adv[im] = adv_img_np[im]
                        adv_found[im] = True
                        c_np[im] = np.minimum(10000., c_np[im] * 1.2)
                    if not adv_found[im]:
                        c_np[im] = np.maximum(0, c_np[im] * 0.95)
            best_adv_big[run] = best_adv
        temp_sum = np.sum((best_adv_big - img) ** 2, axis=(2, 3, 4))
        best_IDs = np.argmin(temp_sum, axis=0)
        best_adv = best_adv_big[best_IDs, range(self.batch_size)]
        print ''

        return best_adv
