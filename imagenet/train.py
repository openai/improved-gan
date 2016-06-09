import tensorflow as tf
from model import save_images
import time
import numpy as np

def train(self, config):
    """Train DCGAN"""

    d_optim = self.d_optim
    g_optim = self.g_optim

    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()
    #self.g_sum = tf.merge_summary([#self.z_sum,
    #    self.d__sum,
    #    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    # self.d_sum = tf.merge_summary([#self.z_sum,
    #     self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Hang onto a copy of z so we can feed the same one every time we store
    # samples to disk for visualization
    assert self.sample_size > self.batch_size
    assert self.sample_size % self.batch_size == 0
    sample_z = []
    steps = self.sample_size // self.batch_size
    assert steps > 0
    sample_zs = []
    for i in xrange(steps):
        cur_zs = self.sess.run(self.zses[0])
        assert all(z.shape[0] == self.batch_size for z in cur_zs)
        sample_zs.append(cur_zs)
    sample_zs = [np.concatenate([batch[i] for batch in sample_zs], axis=0) for i in xrange(len(sample_zs[0]))]
    assert all(sample_z.shape[0] == self.sample_size for sample_z in sample_zs)

    counter = 1

    if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    start_time = time.time()
    print_time = time.time()
    sample_time = time.time()
    save_time = time.time()
    idx = 0
    try:
        while not coord.should_stop():
            idx += 1
            batch_start_time = time.time()

            """
            batch_images = self.images.eval()
            from pylearn2.utils.image import save
            for i in xrange(self.batch_size):
                save("train_image_%d.png" % i, batch_images[i, :, :, :] / 2. + 0.5)
            """


            #for i in xrange(3):
            #    self.sess.run([d_optim], feed_dict=feed_dict)

            _d_optim, _d_sum, \
            _g_optim,  \
            errD_fake, errD_real, errD_class, \
            errG = self.sess.run([d_optim, self.d_sum,
                                            g_optim, # self.g_sum,
                                            self.d_loss_fakes[0],
                                            self.d_loss_reals[0],
                                            self.d_loss_classes[0],
                                            self.g_losses[0]])

            counter += 1
            if time.time() - print_time > 15.:
                print_time = time.time()
                total_time = print_time - start_time
                d_loss = errD_fake + errD_real + errD_class
                sec_per_batch = (print_time - start_time) / (idx + 1.)
                sec_this_batch = print_time - batch_start_time
                print "[Batch %(idx)d] time: %(total_time)4.4f, d_loss: %(d_loss).8f, g_loss: %(errG).8f, d_loss_real: %(errD_real).8f, d_loss_fake: %(errD_fake).8f, d_loss_class: %(errD_class).8f, sec/batch: %(sec_per_batch)4.4f, sec/this batch: %(sec_this_batch)4.4f" \
                    % locals()

            if (idx < 300 and idx % 10 == 0) or time.time() - sample_time > 300:
                sample_time = time.time()
                samples = []
                # generator hard codes the batch size
                for i in xrange(self.sample_size // self.batch_size):
                    feed_dict = {}
                    for z, zv in zip(self.zses[0], sample_zs):
                        if zv.ndim == 2:
                            feed_dict[z] = zv[i*self.batch_size:(i+1)*self.batch_size, :]
                        elif zv.ndim == 4:
                            feed_dict[z] = zv[i*self.batch_size:(i+1)*self.batch_size, :, :, :]
                        else:
                            assert False
                    cur_samples, = self.sess.run(
                        [self.Gs[0]],
                        feed_dict=feed_dict
                    )
                    samples.append(cur_samples)
                samples = np.concatenate(samples, axis=0)
                assert samples.shape[0] == self.sample_size
                save_images(samples, [8, 8],
                            self.sample_dir + '/train_%s.png' % ( idx))


            if time.time() - save_time > 3600:
                save_time = time.time()
                self.save(config.checkpoint_dir, counter)
    except tf.errors.OutOfRangeError:
        print "Done training; epoch limit reached."
    finally:
        coord.request_stop()

    coord.join(threads)
    # sess.close()
