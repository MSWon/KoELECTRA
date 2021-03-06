import tensorflow as tf
import numpy as np
import random

_buffer_size = 2000000
_bucket_size = 10
_thread_num = 16

def get_vocab(vocab_path, isTF=True):
    if isTF:
        vocab_path_tensor = tf.constant(vocab_path)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_path_tensor)
        vocab_dict = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_path_tensor,
            num_oov_buckets=0,
            default_value=1)
    else:
        vocab_dict = {}
        with open(vocab_path, "r") as f:
            for vocab in f:
                vocab_dict[len(vocab_dict)] = vocab.strip()
    return vocab_dict

def tensor_scatter_update(x, indices, updates):
    x_shape = tf.shape(x)
    patch = tf.scatter_nd(indices, updates, x_shape)
    mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), x_shape), 0)
    return tf.where(mask, patch, x)

def scatter_mask_update(tensor, indices, mask_idx):
    updates = tf.fill(tf.shape(indices), mask_idx)
    indices = tf.reshape(indices, [-1, 1])
    return tensor_scatter_update(tensor, indices, updates)

def get_mask_position(org_input_idx, max_len, mask_idx):
    seq_len = tf.shape(org_input_idx)[0]
    max_mask_len = round(max_len * 0.15)

    # Sample 15% of the word
    sample_num_real = tf.to_int32(tf.round(tf.multiply(tf.to_float(seq_len), 0.15)))
    idx_real = tf.range(1, seq_len)
    real_mask = tf.random_shuffle(idx_real)[:sample_num_real]

    sample_num_pad = max_mask_len - sample_num_real
    idx_pad = tf.range(seq_len, max_len)
    pad_mask = tf.random_shuffle(idx_pad)[:sample_num_pad]

    mask_position = tf.concat([real_mask, pad_mask], axis=0)
    # Mask 15% of the word
    masked_input_idx = scatter_mask_update(org_input_idx, real_mask, mask_idx)
    weight_label = tf.concat([tf.ones_like(real_mask), tf.zeros_like(pad_mask)], axis=0)

    output_idx = tf.gather(org_input_idx, real_mask)

    result_dic = {
        "org_input_idx": org_input_idx,
        "masked_input_idx": masked_input_idx,
        "output_idx" : output_idx,
        "seq_len": seq_len,
        "mask_position": mask_position,
        "weight_label": weight_label
    }
    return result_dic

def train_dataset_fn(corpus_path, vocab_path, max_len, mask_idx, batch_size):

  with tf.device("/cpu:0"):
      dataset = tf.data.TextLineDataset(corpus_path)
      tf_vocab = get_vocab(vocab_path)

      dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(_buffer_size))
      # Split & add [CLS] token
      dataset = dataset.map(
            lambda line:
                tf.concat([["[CLS]"], tf.string_split([line]).values], axis=0),
                num_parallel_calls=_thread_num
      )
      # Truncate to max_len
      dataset = dataset.map(
          lambda x:
                tf.cond(tf.greater(tf.shape(x)[0], max_len),
                        lambda: x[:max_len],
                        lambda: x
                        )
      )
      # word -> idx by Lookup table
      dataset = dataset.map(
          lambda x:
            tf_vocab.lookup(x),
            num_parallel_calls=_thread_num
      )
      # Cast to tf.int32
      dataset = dataset.map(lambda x: tf.cast(x, tf.int32))
      # Get mask position
      dataset = dataset.map(
          lambda x:
            get_mask_position(x, max_len, mask_idx),
            num_parallel_calls=_thread_num
      )
      # Padding
      dataset = dataset.padded_batch(
          batch_size,
          {
              "org_input_idx": [max_len],
              "masked_input_idx": [max_len],
              "output_idx": [round(max_len * 0.15)],
              "seq_len": [],
              "mask_position": [round(max_len * 0.15)],
              "weight_label": [round(max_len * 0.15)]
          }
      )
      # Prefetch the next element to improve speed of input pipeline.
      dataset = dataset.prefetch(3)
  return dataset

