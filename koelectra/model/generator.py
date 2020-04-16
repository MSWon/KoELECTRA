from ..utils import model_utils
from .encoder import Encoder
import tensorflow as tf

class Generator(object):
    """ Generator class """
    def __init__(self, hyp_args):
        self.D_hidden_dim = hyp_args['D_hidden_dim']
        self.num_layers = hyp_args['G_num_layers']
        self.num_heads = hyp_args['G_num_heads']
        self.G_hidden_dim = hyp_args['G_hidden_dim']
        self.linear_key_dim = hyp_args['G_linear_key_dim']
        self.linear_value_dim = hyp_args['G_linear_value_dim']
        self.ffn_dim = hyp_args['G_ffn_dim']
        self.dropout = hyp_args['G_dropout']
        self.vocab_size = hyp_args['vocab_size']
        self.activation = hyp_args['G_activation']
        self.layer_norm = model_utils.LayerNormalization(self.D_hidden_dim)
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
            position_emb = model_utils.get_position_encoding(max_seq_length, self.D_hidden_dim)
        # Word Embedding
        with tf.variable_scope("Electra/Embeddings", reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable('Weights', [self.vocab_size, self.D_hidden_dim],
                                                     dtype=tf.float32,
                                                     initializer=tf.random_normal_initializer(
                                                         0., self.D_hidden_dim ** -0.5))
            mask = tf.to_float(tf.not_equal(inputs, 0))
            word_emb = tf.nn.embedding_lookup(self.embedding_weights, inputs)  ## batch_size, length, dim
            word_emb *= tf.expand_dims(mask, -1)  ## zeros out masked positions
            word_emb *= self.D_hidden_dim ** 0.5  ## Scale embedding by the sqrt of the hidden size
        ## Add Word emb & Positional emb
        encoded_inputs = tf.add(word_emb, position_emb)
        ## Linear projection
        encoded_inputs = tf.layers.dense(encoded_inputs, self.G_hidden_dim, name="Generator/embedding_project")

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
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.G_hidden_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              activation=self.activation,
                              isTrain=isTrain)
            ## sequence output, pooled output
            encoder_outputs = encoder.build(encoder_emb_inp, padding_bias)
            self.sequence_output = encoder_outputs
            self.pooled_output =encoder_outputs[:, 0]
            return encoder_outputs

    def build_logits(self, encoder_outputs, mask_position):
        sub_outputs = model_utils.gather_indexes(encoder_outputs, mask_position)  ## batch_size*max_mask, G_hidden_dim
        max_mask = tf.shape(mask_position)[1]
        sub_outputs = tf.reshape(sub_outputs, [-1, max_mask, self.G_hidden_dim])
        
        with tf.variable_scope("Generator/Transform_layer", reuse=tf.AUTO_REUSE):
            transformed_output = tf.layers.dense(sub_outputs, self.D_hidden_dim, activation=self.activation, 
                                                 kernel_initializer=tf.random_normal_initializer(0., self.D_hidden_dim ** -0.5))
            transformed_output = self.layer_norm(transformed_output)

        with tf.variable_scope("Generator/Output_layer", reuse=tf.AUTO_REUSE):
            output_bias = tf.get_variable("output_bias", [self.vocab_size], initializer=tf.zeros_initializer())
            transformed_output = tf.reshape(transformed_output, [-1, self.D_hidden_dim])
            logits = tf.matmul(transformed_output, self.embedding_weights, transpose_b=True)
            logits = tf.reshape(logits, [-1, max_mask, self.vocab_size])
            logits = tf.nn.bias_add(logits, output_bias)
        return logits

    def build_graph(self, input_idx, mask_position):
        ## Encoder
        encoder_outputs = self.build_encoder(input_idx, isTrain=True)
        ## Logits
        logits = self.build_logits(encoder_outputs, mask_position)
        return logits

    def build_loss(self, logits, labels, weight_label):
        """
        :param logits          : (batch_size, max_len*0.15, vocab_size)
        :param labels : (batch_size, max_len*0.15)
        :param weight_label : (batch_size, max_len*0.15)
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # sequence mask for padding
        weight_label = tf.cast(weight_label, dtype = tf.float32)
        loss = tf.reduce_sum(loss * weight_label) / (tf.reduce_sum(weight_label) + 1e-10)
        return loss
