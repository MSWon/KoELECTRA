import tensorflow as tf
from ..utils.model_utils import gelu

class FFN(object):
    """ FFN class (Position-wise Feed-Forward Networks) """
    def __init__(self,
                 w1_dim=2048,
                 w2_dim=512,
                 dropout=0.1,
                 activation="relu"):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout
        self.activation = activation

    def dense_layer(self, inputs, isTrain):
        """Return outputs of the feedforward network.
        Args:
          inputs: tensor with shape [batch_size, length, model_dim]
          padding: the padding values are temporarily removed
            from inputs. The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]
            
        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """           
        output = tf.layers.dense(inputs, self.w1_dim, activation=self.activation)
        if isTrain:
            output = tf.nn.dropout(output, 1.0 - self.dropout)
        output = tf.layers.dense(output, self.w2_dim)
        return output

