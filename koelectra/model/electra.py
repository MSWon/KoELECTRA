from .generator import Generator
from .discriminator import Discriminator
from ..utils.gpu_utils import average_gradients
from ..utils.model_utils import AdamWeightDecayOptimizer, noam_scheme
from ..utils.data_utils import tensor_scatter_update
import tensorflow as tf

class Electra(object):
    """ Electra class """
    def __init__(self, hyp_args):
        self.G_model = Generator(hyp_args)
        self.D_model = Discriminator(hyp_args)
        self.G_weight = hyp_args["G_weight"]
        self.D_weight = hyp_args["D_weight"]
        self.n_gpus = hyp_args["n_gpus"]
        self.temperature = hyp_args["temperature"]

    def build_opt(self, features, d_model, global_step, warmup_steps=10000):
        """
        :param features: train data pipeline
        :param d_model: hidden_dim
        :param global_step: integer
        :param warmup_steps: integer
        :return: train_loss: integer
                 train_opt: optimizer
        """
        # define optimizer
        learning_rate = noam_scheme(d_model, global_step, warmup_steps)

        opt = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        ''' Multi-GPU '''
        train_loss = tf.get_variable('total_loss', [],
                                     initializer=tf.constant_initializer(0.0), trainable=False)

        tower_grads = []
        total_batch = tf.shape(features['org_input_idx'])[0]
        batch_per_gpu = total_batch // self.n_gpus

        with tf.variable_scope(tf.get_variable_scope()):
            for k in range(self.n_gpus):
                with tf.device("/gpu:{}".format(k)):
                    print("Building model tower_{}".format(k + 1))
                    print("Could take few minutes")
                    # calculate the loss for one model replica
                    start = tf.to_int32(batch_per_gpu * k)
                    end = tf.to_int32(batch_per_gpu * (k + 1))
                    org_input_idx = features['org_input_idx'][start:end]
                    G_input_idx = features['masked_input_idx'][start:end]
                    seq_len = features['seq_len'][start:end]
                    output_idx = features['output_idx'][start:end]
                    mask_position = features['mask_position'][start:end]
                    weight_label = features['weight_label'][start:end]

                    G_logits = self.G_model.build_graph(G_input_idx, mask_position) # batch_size, mask_len, vocab_size
                    G_loss = self.G_model.build_loss(G_logits, output_idx, weight_label)
                    G_logits_ = tf.stop_gradient(tf.nn.softmax(G_logits/ self.temperature))
                    G_infer_idx = tf.argmax(G_logits_, axis=-1)   # batch_size, mask_len
                    G_infer_idx = tf.cast(G_infer_idx, tf.int32)

                    indices = mask_position + tf.range(0, batch_per_gpu*tf.shape(G_input_idx)[1], tf.shape(G_input_idx)[1])[:,None]
                    indices = tf.reshape(indices, [-1,1])
                    input_idx_flatten = tf.reshape(G_input_idx, [-1])  # batch_size * seq_len
                    G_infer_idx_flatten = tf.reshape(G_infer_idx, [-1]) # batch_size * mask_len

                    D_input_idx = tensor_scatter_update(input_idx_flatten, indices, G_infer_idx_flatten) # batch_size * seq_len
                    D_input_idx = tf.reshape(D_input_idx, [-1, tf.shape(G_input_idx)[1]])  # batch_size , seq_len
                    # exclude [CLS] where it's index is 0
                    D_logits = self.D_model.build_graph(D_input_idx[:,1:])
                    # compare original input with Discriminator's input from index 1 (exclude [CLS] where it's index is 0)
                    D_labels = tf.cast(tf.equal(org_input_idx[:,1:], D_input_idx[:,1:]), tf.int32)
                    D_loss = self.D_model.build_loss(D_logits, D_labels, seq_len-1)

                    loss = self.G_weight * G_loss + self.D_weight * D_loss
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    grads_and_vars = opt.compute_gradients(loss)
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=1.0), var) for (grad, var) in grads_and_vars]
                    tower_grads.append(grads_and_vars)
                    train_loss += loss / self.n_gpus

        grads = average_gradients(tower_grads)
        train_opt = opt.apply_gradients(grads, global_step=global_step)
        return train_loss, train_opt
