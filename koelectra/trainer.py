import tensorflow as tf
import numpy as np
import os
from koelectra.utils.data_utils import train_dataset_fn
from koelectra.model.electra import Electra

class Trainer(object):
    """ Trainer class """
    def __init__(self, hyp_args):
        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        path = uppath(__file__, 1)

        self.corpus_path = os.path.join(path,hyp_args["corpus_path"])
        self.vocab_path = os.path.join(path,hyp_args["vocab_path"])
        self.max_init_word_len = hyp_args["max_init_word_len"]
        self.max_converge_word_len = hyp_args["max_converge_word_len"]
        self.vocab_size = hyp_args["vocab_size"]
        self.init_batch_size = hyp_args["init_batch_size"]
        self.converge_batch_size = hyp_args["converge_batch_size"]
        self.model_path = hyp_args["model_path"]
        self.training_steps = hyp_args["training_steps"]
        self.converge_steps = hyp_args["converge_steps"]
        self.mask_idx = hyp_args["mask_idx"]
        self.cls_idx = hyp_args["cls_idx"]

        train_init_dataset = train_dataset_fn(self.corpus_path,self.vocab_path,
                                              self.max_init_word_len, self.mask_idx, self.cls_idx, self.init_batch_size)

        train_converge_dataset = train_dataset_fn(self.corpus_path,self.vocab_path,
                                              self.max_converge_word_len, self.mask_idx, self.cls_idx, self.converge_batch_size)

        iters = tf.data.Iterator.from_structure(train_init_dataset.output_types,
                                                train_init_dataset.output_shapes)
        features = iters.get_next()

        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_init_dataset)
        self.train_converge_op = iters.make_initializer(train_converge_dataset)

        print("Now building model")
        model = Electra(hyp_args)
        global_step = tf.train.get_or_create_global_step()

        self.train_loss, self.train_opt = model.build_opt(features, hyp_args["hidden_dim"],
                                                          global_step, hyp_args["warmup_step"])

        ## for tensorboard
        self.train_loss_graph = tf.placeholder(shape=None, dtype=tf.float32)
        self.train_ppl_graph = tf.placeholder(shape=None, dtype=tf.float32)
        print("Done")

    def train(self):
        """
        :param training_steps: integer
        :return: None
        """
        print("Now training")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("./model")
        summary_loss = tf.summary.scalar("loss", self.train_loss_graph)
        summary_ppl = tf.summary.scalar("perplexity", self.train_ppl_graph)
        merged = tf.summary.merge([summary_loss, summary_ppl])
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, "./model/" + self.model_path)
                print("Model loaded!")

            sess.run(self.train_init_op)
            sess.run(tf.tables_initializer())
            n_train_step = 0
            writer = tf.summary.FileWriter('./tensorboard/graph', sess.graph)

            for step in range(self.training_steps):
                if step == self.converge_steps:
                    sess.run(self.train_converge_op)
                n_train_step += 1
                batch_train_loss, _ = sess.run([self.train_loss,
                                                self.train_opt])
                batch_train_ppl = np.exp(batch_train_loss)

                print("step : {:06d} train_loss : {:.6f} perplexity : {:.6f}".format(step + 1, batch_train_loss, batch_train_ppl))

                if step % 100 == 0 and step > 0:
                    summary = sess.run(merged,
                                       feed_dict={self.train_loss_graph: batch_train_loss,
                                                  self.train_ppl_graph: batch_train_ppl})
                    writer.add_summary(summary, step)

                if step % 10000 == 0 and step > 0:
                    saver.save(sess, "./model/" + self.model_path)
