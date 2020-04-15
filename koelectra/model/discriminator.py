from ..utils import model_utils
from .encoder import Encoder
import tensorflow as tf

class Discriminator(object):
    """ Discriminator class """
    def __init__(self, hyp_args):
        self.num_layers = hyp_args['D_num_layers']
        self.num_heads = hyp_args['D_num_heads']
        self.hidden_dim = hyp_args['D_hidden_dim']
        self.linear_key_dim = hyp_args['D_linear_key_dim']
        self.linear_value_dim = hyp_args['D_linear_value_dim']
        self.ffn_dim = hyp_args['D_ffn_dim']
        self.dropout = hyp_args['D_dropout']
        self.vocab_size = hyp_args['vocab_size']
        self.activation = hyp_args['D_activation']
        self.layer_norm = model_utils.LayerNormalization(self.hidden_dim)
        if self.activation == "gelu":
            self.activation = model_utils.gelu

    def build_embed(self, inputs, isTrain):
        """
        :param inputs: (batch_size, max_len)
        :param isTrain: boolean (True/False)
        :return: (batch_size, max_len, emb_dim)
        """
        max_seq_length = tf.shape(inputs)[1]
        # Positional Encoding
        with tf.variable_scope("Electra/Positional-encoding", reuse=tf.AUTO_REUSE):
            position_emb = model_utils.get_position_encoding(max_seq_length, self.hidden_dim)
        # Word Embedding
        with tf.variable_scope("Electra/Embeddings", reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable('Weights', [self.vocab_size, self.hidden_dim],
                                                     dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(
                                                         0., self.hidden_dim ** -0.5))
            mask = tf.to_float(tf.not_equal(inputs, 0))
            word_emb = tf.nn.embedding_lookup(self.embedding_weights, inputs)  ## batch_size, length, dim
            word_emb *= tf.expand_dims(mask, -1)  ## zeros out masked positions
            word_emb *= self.hidden_dim ** 0.5  ## Scale embedding by the sqrt of the hidden size
        ## Add Word emb & Positional emb
        encoded_inputs = tf.add(word_emb, position_emb)
        if isTrain:
            return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)
        else:
            return encoded_inputs

    def build_encoder(self, enc_input_idx, isTrain):
        ## enc_input_idx : (batch_size, enc_len)
        """
        :param enc_input_idx: (batch_size, enc_len)
        :param isTrain: boolean (True/False)
        :return: (batch_size, enc_len, hidden_dim)
        """
        padding_bias = model_utils.get_padding_bias(enc_input_idx)
        encoder_emb_inp = self.build_embed(enc_input_idx, isTrain)
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              activation=self.activation,
                              isTrain=isTrain)

            return encoder.build(encoder_emb_inp, padding_bias)

    def build_logits(self, encoder_outputs):
        with tf.variable_scope("Discriminator/Transform_layer", reuse=tf.AUTO_REUSE):
            transformed_output = tf.layers.dense(sub_outputs, self.hidden_dim, activation=self.activation)
            transformed_output = self.layer_norm(transformed_output)

        with tf.variable_scope("Discriminator/Output_layer", reuse=tf.AUTO_REUSE):
            logits = tf.squeeze(tf.layers.dense(transformed_output, 1), -1)
        return logits

    def build_graph(self, input_idx):
        ## Encoder
        encoder_outputs = self.build_encoder(input_idx, isTrain=True)
        ## Logits
        logits = self.build_logits(encoder_outputs)
        return logits

    def build_loss(self, logits, labels, seq_len):
        """
        :param logits : (batch_size, max_len)
        :param labels : (batch_size, max_len)
        :param weight_label : (batch_size,)
        """
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        weight_label = tf.cast(tf.sequence_mask(seq_len, maxlen=tf.shape(labels)[1]), tf.float32)
        # sequence mask for padding
        loss = tf.reduce_sum(loss * weight_label) / (tf.reduce_sum(weight_label) + 1e-10)
        return loss
