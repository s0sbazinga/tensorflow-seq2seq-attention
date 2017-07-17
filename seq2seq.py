import os
import math
import time
import threading
from  datetime import datetime
import six
from six.moves import xrange
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.client import timeline
from tensorflow.contrib.rnn import CompiledWrapper
from tensorflow.python import debug as tf_debug

from attention import AttentionDecoder, dynamic_attention_rnn


UNK_IDX = 2
START = "<s>"
END = "<e>"

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 5e-4, "Learning rate (default: 5e-4)")
tf.flags.DEFINE_float("cell_size", 1024, "Number of hidden units of encoder/decoder rnn cell (default: 1024)")
tf.flags.DEFINE_integer("embedding_dim", 1024, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("max_seq_len", 80, "Max sequence length (default: 80)")
tf.flags.DEFINE_integer("num_enc_layers", 2, "Encoder layers (default: 1)")
tf.flags.DEFINE_integer("num_dec_layers", 2, "Decoder layers (default: 1)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 8e-5, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_boolean("use_coverage", True, "Coverage model (default: True)")
tf.flags.DEFINE_float("init_scale", 1e-2, "Weight initialization scale value (default: 1e-2)")
tf.flags.DEFINE_float("embedding_init_scale", 1e-2, "Weight initialization scale value (default: 1e-2)")

# Training parameters
tf.flags.DEFINE_integer("num_gpus", 1, "Number of GPUs (default: 4)")
tf.flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 80)")
tf.flags.DEFINE_integer("valid_batch_size", 512, "Dev batch Size (default: 512)")
tf.flags.DEFINE_integer("clip_gradients", 25, "Clip gradients (default: 25)")
tf.flags.DEFINE_integer("moving_average_decay", 0.9999, "Moving average decay (default: 0.9999)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("show_train_stat_every", 10, "Show training stats after this many steps (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 200, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS


def read_vocab(src_vocab_file, trg_vocab_file):
  src_dict = {}
  for line_count, line in enumerate(open(src_vocab_file, "r")):
    src_dict[line.strip()] = line_count
  trg_dict = {}
  for line_count, line in enumerate(open(trg_vocab_file, "r")):
    trg_dict[line.strip()] = line_count
  return src_dict, trg_dict


def pad_batch(batch, max_len):
  x, y, z = [np.zeros([len(batch), max_len], dtype=np.int32) + UNK_IDX for _ in range(3)]
  x_lens, y_lens = [[[len(s)] for s in seqs] for seqs in zip(*batch)[:-1]]
  for i, (a, b, c) in enumerate(batch):
      x[i:, :len(a)] = a
      y[i:, :len(b)] = b
      z[i:, :len(c)] = c
  return x, x_lens, y, y_lens, z


class DataGenerator(object):
  def __init__(self, data_dir, data_filename, batch_size, num_epochs,
         src_vocab, trg_vocab=None, max_seq_len=80, cap_rate=10000):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.data_dir = data_dir
    self.data_filename = data_filename
    self.queue_capacity =  cap_rate * batch_size

  def build_data_generator(self):
    mode = self.trg_vocab is not None    # training

    def _get_ids(s, dictionary):
      words = s.strip().split()
      return [dictionary[START]] + \
           [dictionary.get(w, UNK_IDX) for w in words] + \
           [dictionary[END]]

    data_file = os.paht.join(self.data_dir, self.data_filename)
    with open(self.data_file, 'r') as fd:
      for line_count, line in enumerate(fd):
        # if line_count >= 128:
        #     break
        line_split = line.strip().split('\t')
        if mode and len(line_split) != 2:
          continue
        src_seq = line_split[0]  # one source sequence
        src_ids = _get_ids(src_seq, self.src_vocab)
        src_ids.reverse()

        if mode:
          trg_seq = line_split[1]  # one target sequence
          trg_words = trg_seq.split()
          trg_ids = [self.trg_vocab.get(w, UNK_IDX) for w in trg_words]

          # remove sequence whose length > 80 in training mode
          if len(src_ids) > FLAGS.max_seq_len or len(trg_ids)+1 > FLAGS.max_seq_len:
            continue
          trg_ids_next = trg_ids + [self.trg_vocab[END]]
          trg_ids = [self.trg_vocab[START]] + trg_ids
          yield src_ids, trg_ids, trg_ids_next
        else:
          yield src_ids, [line_count]

  def _build_input_graph(self, mode):
    with tf.variable_scope("%s_input_graph" % mode):
      max_len = self.max_len
      self.src_batch = tf.placeholder(tf.int32, [None], name="x_src")
      self.src_len_batch = tf.placeholder(tf.int32, [None], name="x_src_len")
      self.trg_batch = tf.placeholder(tf.int32, [None], name="x_trg")
      self.trg_len_batch = tf.placeholder(tf.int32, [None], name="x_trg_len")
      self.y_trg_batch = tf.placeholder(tf.int32, [None], name="y_trg")
      self.queue = tf.PaddingFIFOQueue(capacity=self.queue_capacity,
        dtypes=[tf.int32] * 5, shapes=[[None] for _ in range(5)])
      self.enqueue_op = self.queue.enqueue([self.src_batch, self.src_len_batch,
        self.trg_batch, self.trg_len_batch, self.y_trg_batch])
      self.dequeue_op = self.queue.dequeue()

  def _enqueue(self, data_batch, sess):
    x, y, z = data_batch
    x_lens = [len(x)]
    y_lens = [len(y)]
    # x, x_lens, y, y_lens, z = pad_batch(data_batch, self.max_len)
    feed_dict = {
      self.src_batch: x,
      self.src_len_batch: x_lens,
      self.trg_batch: y,
      self.trg_len_batch: y_lens,
      self.y_trg_batch: z,
    }
    size_op = self.queue.size()
    sess.run([self.enqueue_op], feed_dict=feed_dict)

  def _run(self, sess, coord, is_eval=False):
    # with tf.variable_scope("input_fn"):
    try:
      for i in xrange(self.num_epochs):
        if coord and coord.should_stop():
          break
        end = False
        datagen = self.build_data_generator()
        while True:
          if coord and coord.should_stop():
            break
          data_batch = []
          for _ in range(self.batch_size):
            try:
              data = datagen.next()
              self._enqueue(data, sess)
              # data_batch.append(datagen.next())
            except StopIteration:
              end = True
          if not data_batch:
            if end:
              break
            else:
              continue
    except Exception as e:
      if not is_eval:
        print("Error when fetching data...", e)
        sess.run(self.queue.close(cancel_pending_enqueues=True))
        coord.request_stop(e)
      return
    finally:
      mode = "dev" if is_eval else "train"
      print("Data loading for %s finished." % mode)
      if not is_eval:
        sess.run(self.queue.close())

  def create_input_fn(self, sess, coord, min_queue_size, is_eval=False):
    mode = "train" if not is_eval else "dev"
    with tf.variable_scope("%s_input" % mode):
      self._build_input_graph(mode)
      enqueue_thread = threading.Thread(target=self._run,
        args=[sess, coord, is_eval])
      enqueue_thread.daemon = True
      def input_fn():
        shuffle = True if not is_eval else False
        with tf.variable_scope("%s_input_fn" % mode):
          min_after_dequeue = min_queue_size
          capacity = min_after_dequeue + 3 * FLAGS.batch_size
          if shuffle:
            # shuffled_data_batch = tf.train.shuffle_batch(self.dequeue_op,
            #   batch_size=FLAGS.batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,
            #   num_threads=16, allow_smaller_final_batch=False)
            _, data_batch = tf.contrib.training.bucket_by_sequence_length(
              self.dequeue_op[1], self.dequeue_op, batch_size=FLAGS.batch_size, num_threads=16,
              bucket_boundaries=[10, 20, 30, 40, 50, 60, 70, 80], dynamic_pad=True,
              capacity=3 * FLAGS.batch_size, allow_smaller_final_batch=True)
          else:
            data_batch = tf.train.batch(self.dequeue_op,
              batch_size=FLAGS.batch_size, capacity=capacity,
              num_threads=4, allow_smaller_final_batch=False)
        return data_batch
    return self.queue, enqueue_thread, input_fn

  def create_input_fn_new(self, capacity=80, num_datashards=20, is_training=True):
    buckets = [10, 20, 30, 40, 50, 60, 70, 80]

    capacity *= num_datashards
    path = os.path.join(self.data_dir, self.data_filename)
    data_file_pattern = ("%s-train*" % path) if is_training else ("%s-dev*" % path)

    examples = self.pipeline(data_file_pattern, capacity, is_training)
    examples.update({
        "inputs_len": tf.shape(examples["inputs"])[0],
        "targets_len": tf.shape(examples["targets"])[0]
    })

    bucket_capacities = [2 * self.batch_size for _ in range(len(buckets) + 1)]

    def input_fn():
        with tf.name_scope("batch_examples"), tf.device("/cpu:0"):
            (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
                examples["inputs_len"],
                examples,
                self.batch_size,
                buckets,
                capacity=2 * FLAGS.num_gpus,
                bucket_capacities=bucket_capacities,
                dynamic_pad=True,
                allow_smaller_final_batch=True)
            return outputs

    return input_fn

  def pipeline(self, data_file_pattern, capacity, is_training=True):
      data_fields = {
          "inputs": tf.VarLenFeature(tf.int64),
          "targets": tf.VarLenFeature(tf.int64)
      }
      data_sources = [data_file_pattern]

      def _gen_examples():
          with tf.name_scope("examples_queue"), tf.device("/cpu:0"):
              # Read serialized examples using slim parallel_reader.
              num_epochs = self.num_epochs if is_training else 1
              data_files = tf.contrib.slim.parallel_reader.get_data_files(data_sources)
              num_readers = min(4 if is_training else 1, len(data_files))
              _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
                  data_sources,
                  tf.TFRecordReader,
                  num_epochs=num_epochs,
                  shuffle=is_training,
                  capacity=2 * capacity,
                  min_after_dequeue=capacity,
                  num_readers=num_readers)

              data_items_to_decoders = {
                  field: tf.contrib.slim.tfexample_decoder.Tensor(field)
                  for field in data_fields
              }

              decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
                  data_fields, data_items_to_decoders)
              
              data_items_to_decode = list(data_items_to_decoders)

              decoded = decoder.decode(example_serialized, items=data_items_to_decode)
              return {
                  field: tensor
                  for (field, tensor) in zip(data_items_to_decode, decoded)
              }

      examples = _gen_examples()
      # We do not want int64s as they do are not supported on GPUs.
      return {k: tf.to_int32(v) for (k, v) in six.iteritems(examples)}


def batch_iter(data, batch_size, num_epochs, shuffle=True):
  data = np.array(data)
  data_len = len(data)
  for epoch in range(num_epochs):
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_len))
      shuffle_data = data[shuffle_indices]
    else:
      shuffle_data = data
    num_batches = int(data_len-1)/batch_size + 1
    for batch_num in range(num_batches):
      start_index = batch_num
      end_index = min((batch_num + 1) * batch_size, data_len)
      yield shuffle_data[start_index:end_index]


def init_std(out_dim):
  assert out_dim > 0, "Wrong output dim."
  return 1. / math.sqrt(out_dim)

def get_rnn_cell(cell_size, inp_dim=None, use_lstm=False, use_residual=False,
                 use_dropout=False, dropout_keep_prob=1.):
  if inp_dim is None:
    inp_dim = cell_size
  
  # initializer = tf.random_normal_initializer(mean=0, stddev=init_std(inp_dim))
  std = init_std(inp_dim)
  initializer = tf.random_uniform_initializer(-std, std)
  bias_initializer = tf.constant_initializer(0.0, dtype=tf.float32)

  if use_lstm:
    cell = rnn_cell.LSTMCell(cell_size, kernel_initializer=initializer,
                             bias_initializer=bias_initializer, state_is_tuple=True)
  else:
    cell = rnn_cell.GRUCell(cell_size, kernel_initializer=initializer,
                            bias_initializer=bias_initializer)

  if use_residual:
    cell = rnn_cell.ResidualWrapper(cell)

  if use_dropout:
    cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
  return cell


def create_learning_rate_decay_fn(decay_type,
                  decay_steps,
                  decay_rate,
                  start_decay_at=0,
                  stop_decay_at=1e9,
                  min_learning_rate=None,
                  staircase=False):
  """Creates a function that decays the learning rate.

  Args:
  decay_steps: How often to apply decay.
  decay_rate: A Python number. The decay rate.
  start_decay_at: Don't decay before this step
  stop_decay_at: Don't decay after this step
  min_learning_rate: Don't decay below this number
  decay_type: A decay function name defined in `tf.train`
  staircase: Whether to apply decay in a discrete staircase,
    as opposed to continuous, fashion.

  Returns:
  A function that takes (learning_rate, global_step) as inputs
  and returns the learning rate for the given step.
  Returns `None` if decay_type is empty or None.
  """
  if decay_type is None or decay_type == "":
    return None

  start_decay_at = tf.to_int32(start_decay_at)
  stop_decay_at = tf.to_int32(stop_decay_at)

  def decay_fn(learning_rate, global_step):
    """The computed learning rate decay function.
    """
    global_step = tf.to_int32(global_step)

    decay_type_fn = getattr(tf.train, decay_type)
    decayed_learning_rate = decay_type_fn(
      learning_rate=learning_rate,
      global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=staircase,
      name="decayed_learning_rate")

    final_lr = tf.train.piecewise_constant(
      x=global_step,
      boundaries=[start_decay_at],
      values=[learning_rate, decayed_learning_rate])

    if min_learning_rate:
      final_lr = tf.maximum(final_lr, min_learning_rate)

    return final_lr

  return decay_fn


class Seq2SeqModel(object):
  monitor = None
  def __init__(self, params, name="att_seq2seq"):
    self.params = params
    dropout_keep_prob = FLAGS.dropout_keep_prob
    with tf.name_scope("dropout"):
      self.dropout_keep_prob = tf.get_variable("dropout_keep_prob", [],
        initializer=tf.constant_initializer(dropout_keep_prob), trainable=False)
    self.num_enc_layers = FLAGS.num_enc_layers
    self.num_dec_layers = FLAGS.num_dec_layers
    self.cell_size = FLAGS.cell_size
    self.use_coverage = FLAGS.use_coverage

  def encode(self, x_src, x_src_len):
    with tf.variable_scope("encoder") as vs:
      with tf.device("/cpu:0"):
        src_init_std = init_std(self.params["src_vocab_size"])
        src_emb_w = tf.get_variable(
          name="W",
          shape=[self.params["src_vocab_size"], FLAGS.embedding_dim],
          # initializer=tf.random_normal_initializer(mean=0, stddev=src_init_std))
          initializer=tf.random_uniform_initializer(
           -src_init_std, src_init_std))
      src_embeded = tf.nn.embedding_lookup(src_emb_w, x_src)
      if self.num_enc_layers > 2:
        cell = rnn_cell.MultiRNNCell([get_rnn_cell(self.cell_size)
                                      for _ in range(self.num_enc_layers - 1)])
      else:
        cell = get_rnn_cell(self.cell_size, inp_dim=self.cell_size)
      bidir_outputs, bidir_state = tf.nn.bidirectional_dynamic_rnn(
                              cell_fw=get_rnn_cell(self.cell_size),
                              cell_bw=get_rnn_cell(self.cell_size),
                              inputs=src_embeded,
                              sequence_length=x_src_len,
                              dtype=tf.float32)
      bidir_outputs = tf.concat(bidir_outputs, axis=-1)
      outputs, state = tf.nn.dynamic_rnn(
                              cell=cell,
                              inputs=bidir_outputs,
                              sequence_length=x_src_len,
                              dtype=tf.float32)
      if self.use_coverage:
        with tf.name_scope("coverage"):
          coverage_w = tf.get_variable(
                              name="coverage_w",
                              shape=[self.cell_size, 1],
                              initializer=tf.random_normal_initializer(0, init_std(self.cell_size)))
                              # initializer=tf.random_uniform_initializer(
                              #  -FLAGS.init_scale, FLAGS.init_scale))
          o = tf.reshape(outputs, [-1, self.cell_size])
          encoded_fert = tf.nn.sigmoid(tf.matmul(o, coverage_w))
          encoded_fertility = tf.reshape(encoded_fert, [-1, self.max_seq_len])
          with tf.control_dependencies([encoded_fertility]):
            encoded_fert_init = tf.zeros_like(encoded_fertility)
        return outputs, state, encoded_fert_init, encoded_fertility
      else:
        return outputs, state
    
  def decode(self, x_trg, x_trg_len, context, x_src_len, 
             init_state=None, init_memory=None):
    with tf.variable_scope("decoder") as vs:
      with tf.device("/cpu:0"):
        trg_init_std = init_std(self.params["trg_vocab_size"])
        trg_emb_w = tf.get_variable(
            name="W",
            shape=[self.params["trg_vocab_size"], FLAGS.embedding_dim],
            initializer=tf.random_normal_initializer(mean=0, stddev=trg_init_std))
            # initializer=tf.random_uniform_initializer(
            #  -FLAGS.embedding_init_scale, FLAGS.embedding_init_scale))
      trg_embeded = tf.nn.embedding_lookup(trg_emb_w, x_trg)
      if self.num_dec_layers > 1:
        cell = rnn_cell.MultiRNNCell([get_rnn_cell(self.cell_size)
                            for _ in range(self.num_dec_layers)])
      else:
        cell = get_rnn_cell(self.cell_size)

      logit_fn = lambda state: self.logit_step(state)
      decoder = AttentionDecoder(embeddings=trg_emb_w, logit_fn=logit_fn)
      if self.use_coverage:
        decoder_outputs, extra_outputs, state = dynamic_attention_rnn(
                              decoder=decoder,
                              cell=cell,
                              inputs=trg_embeded,
                              context=context,
                              att_sequence_length=x_src_len,
                              sequence_length=x_trg_len,
                              use_coverage=True,
                              dtype=tf.float32)
        # extra_outputs = logit_fn(decoder_outputs)
      else:
        decoder_outputs, extra_outputs, state = dynamic_attention_rnn(
                              decoder=decoder,
                              cell=cell,
                              inputs=trg_embeded,
                              context=context,
                              att_sequence_length=x_src_len,
                              sequence_length=x_trg_len,
                              use_coverage=False,
                              dtype=tf.float32)
      return extra_outputs, state

  def decode_infer(self, context, x_src_len, 
                   init_state=None, init_memory=None):
    with tf.variable_scope("decoder") as vs:
      trg_emb_w = tf.get_variable(
          name="W",
          shape=[self.params["trg_vocab_size"], FLAGS.embedding_dim])
          # initializer=tf.random_uniform_initializer(
          #  -FLAGS.embedding_init_scale, FLAGS.embedding_init_scale))

      infer_params = {
        "beam_width": 3, # self.params["inference.beam_search.beam_width"],
        "eos_token": 1,
        "vocab_size": self.params["trg_vocab_size"],
      }
      start_token = 0
      x_trg = tf.fill([infer_params["beam_width"]], start_token)
      # (B, T, D)
      trg_embeded = tf.nn.embedding_lookup(trg_emb_w, tf.expand_dims(x_trg, 1))
      if self.num_dec_layers > 1:
        cell = rnn_cell.MultiRNNCell([get_rnn_cell(self.cell_size)
                            for _ in range(self.num_dec_layers)])
      else:
        cell = get_rnn_cell(self.cell_size)

      logit_fn = lambda state: self.logit_step(state)
      decoder = AttentionDecoder(embeddings=trg_emb_w, logit_fn=logit_fn,
                     is_infer=True, infer_params=infer_params)
      if self.use_coverage:
        decoder_outputs, extra_outputs, state = dynamic_attention_rnn(
                              decoder=decoder,
                              cell=cell,
                              inputs=trg_embeded,
                              context=context,
                              att_sequence_length=x_src_len,
                              sequence_length=None,
                              use_coverage=True,
                              dtype=tf.float32)
      else:
        decoder_outputs, extra_outputs, state = dynamic_attention_rnn(
                              decoder=decoder,
                              cell=cell,
                              inputs=trg_embeded,
                              context=context,
                              att_sequence_length=x_src_len,
                              sequence_length=None,
                              use_coverage=False,
                              dtype=tf.float32)
      return extra_outputs, state

  def _clip_gradients(self, grads_and_vars):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(
      gradients, FLAGS.clip_gradients)
    return list(zip(clipped_gradients, variables))

  def _create_lr_decay_fn(self, decay_steps):
    learning_rate_decay_fn = create_learning_rate_decay_fn(
      decay_type=self.params["optimizer.lr_decay_type"] or None,
      decay_steps=decay_steps,
      decay_rate=self.params["optimizer.lr_decay_rate"],
      start_decay_at=self.params["optimizer.lr_start_decay_at"],
      stop_decay_at=self.params["optimizer.lr_stop_decay_at"],
      min_learning_rate=self.params["optimizer.lr_min_learning_rate"],
      staircase=self.params["optimizer.lr_staircase"])
    return learning_rate_decay_fn

  def _build_train_op(self, loss):
    """Creates the training operation"""
    learning_rate_decay_fn = self._create_lr_decay_fn()
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=self.global_step,
      learning_rate=FLAGS.learning_rate,
      learning_rate_decay_fn=learning_rate_decay_fn,
      clip_gradients=self._clip_gradients,
      optimizer=optimizer,
      summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return train_op, self.global_step

  def logit_step(self, cell_outputs):
    # Optional
    # softmax_input = tf.contrib.layers.fully_connected(
    #     inputs=cell_outputs,
    #     num_outputs=self.cell_size,
    #     activation_fn=tf.nn.tanh,
    #     scope="softmax_input")
    logits = tf.contrib.layers.fully_connected(
      inputs=cell_outputs,
      num_outputs=self.params["trg_vocab_size"],
      activation_fn=None,
      weights_initializer=tf.random_normal_initializer(0, init_std(self.cell_size)),
      scope="logits")

    return logits

  def _build_decode_fn(self, is_infer):
    def decode_fn(*args, **kwargs):
      return (self.decode_infer(*args[2:], **kwargs)
           if is_infer else self.decode(*args, **kwargs))
    return decode_fn

  def inference(self, features, is_infer=False):
    # build seq2seq model
    x_src, x_src_len, x_trg, x_trg_len = features

    decode = self._build_decode_fn(is_infer)
    if self.use_coverage:
      encoder_context, _, encoded_fert_init, encoded_fertility = self.encode(x_src, x_src_len)
      group_context = [encoded_fert_init, encoded_fertility, encoder_context]
      decoder_outputs, _ = decode(x_trg, x_trg_len, group_context, x_src_len)
    else:
      encoder_context, _ = self.encode(x_src, x_src_len)
      decoder_outputs, _ = decode(x_trg, x_trg_len, encoder_context, x_src_len)

    return decoder_outputs, encoder_context

  def loss(self, logits, labels):
    x_trg_len, y_trg = labels
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y_trg)
    loss_mask = tf.sequence_mask(
        tf.to_int32(x_trg_len), tf.to_int32(tf.shape(y_trg)[1]))

    # soft_logits = tf.nn.softmax(logits)
    # one_hot_labels = tf.to_float(tf.one_hot(tf.to_int32(y_trg), self.params["trg_vocab_size"]))
    # one_hot_loss = -tf.log(one_hot_labels * soft_logits + (1.0 - one_hot_labels))
    # one_hot_loss = one_hot_loss * tf.to_float(tf.expand_dims(loss_mask, -1))
    # one_hot_loss = tf.reduce_sum(one_hot_loss) # / float(FLAGS.num_gpus * FLAGS.batch_size)
    # loss = tf.reduce_sum(losses * tf.to_float(loss_mask)) / tf.to_float(
    #     tf.reduce_sum(x_trg_len))
    loss = tf.reduce_sum(losses * tf.to_float(loss_mask)) # / float(FLAGS.num_gpus * FLAGS.batch_size)
    # Seq2SeqModel.monitor = tf.gradients(one_hot_loss, logits)
    weight_decay_loss = 0.0
    for v in tf.trainable_variables():
      v_size = int(np.prod(np.array(v.shape.as_list())))
      if len(v.shape.as_list()) > 1:
        # Add weight regularization if set and the weight is not a bias (dim>1).
        with tf.device(v._ref().device):
          v_loss = tf.nn.l2_loss(v)
        weight_decay_loss += v_loss
    weight_decay_loss = FLAGS.l2_reg_lambda * weight_decay_loss
    tf.summary.scalar("weight_decay_loss", weight_decay_loss)

    total_loss = loss + weight_decay_loss
    return total_loss, one_hot_loss

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    COPY FROM OFFICIAL CIFAR10 EXAMPLE!!!
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)
      # if not grads:
      #   continue
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def _format_input(self, input_batch):
    targets = input_batch["targets"]
    features = [input_batch["inputs"], input_batch["inputs_len"],
                targets[:, :-1], input_batch["targets_len"]]

    labels = [input_batch["targets_len"], targets[:, 1:]]

    with tf.name_scope("input_stats"):
      for (k, v) in six.iteritems(input_batch):
        if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
          # tf.summary.scalar("%s_batch" % k, tf.shape(v)[0])
          tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
          nonpadding = tf.to_float(tf.not_equal(v, 0))
          tf.summary.scalar("%s_nonpadding_tokens" % k,
                            tf.reduce_sum(nonpadding))
          tf.summary.scalar("%s_nonpadding_fraction" % k,
                            tf.reduce_mean(nonpadding))

    return features, labels

  def build_model(self, input_fn=None):
    # single gpu version
    with tf.name_scope("seq2seq"), tf.device('/gpu:0'):
      # inputs = input_fn()
      self.src_batch = tf.placeholder(tf.int32, [None, None], name="x_src1")
      self.src_len_batch = tf.placeholder(tf.int32, [None, 1], name="x_src_len1")
      self.trg_batch = tf.placeholder(tf.int32, [None, None], name="x_trg1")
      self.trg_len_batch = tf.placeholder(tf.int32, [None, 1], name="x_trg_len1")
      self.y_trg_batch = tf.placeholder(tf.int32, [None, None], name="y_trg1")
      input_batch = [self.src_batch, self.src_len_batch,
        self.trg_batch, self.trg_len_batch, self.y_trg_batch]
      self.max_seq_len = tf.shape(self.src_batch)[-1]
      # input_batch = sess.run(inputs)

      features, labels = self._format_input(input_batch)
      logits = self.inference(features)
      extra = features

      loss = self.loss(logits, labels)

      train_op, step = self._build_train_op(loss)
    return (train_op, step, loss, extra)

  def build_eval_model(self, input_fn):
    with tf.variable_scope(tf.get_variable_scope()):
      inputs = input_fn()
      features, labels = self._format_input(inputs)
      self.max_seq_len = tf.shape(features[0])[-1]
      logits = self.inference(features)
      loss = self.loss(logits, labels)

      mask = tf.sequence_mask(
          tf.to_int32(labels[0]), tf.to_int32(tf.shape(labels[1])[1]))
      num_tokens = tf.reduce_sum(labels[0] * tf.to_int32(mask))
      tf.summary.scalar('dev_loss', loss)
      summary_op = tf.summary.merge_all()
    return logits, loss, summary_op, num_tokens

  def build_model_multi_gpu(self, input_fn):
    # with tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [],
                     initializer=tf.constant_initializer(0), 
                     trainable=False)
    learning_rate = FLAGS.learning_rate
    lr = tf.get_variable('learning_rate', [],
               initializer=tf.constant_initializer(learning_rate),
               trainable=False)

    num_batches_per_epoch = (self.params["num_examples_per_epoch_for_train"] / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch)
    learning_rate_decay_fn = self._create_lr_decay_fn(decay_steps)
    lr = learning_rate_decay_fn(lr, global_step)

    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

    tower_grads = []
    # inputs = input_fn()
    # batch_queue = prefetch_queue(
    #   inputs, shapes=None, capacity=2 * FLAGS.num_gpus)
    # tf.get_variable_scope().set_initializer(tf.random_normal_initializer(
    #       mean=0, stddev=FLAGS.init_scale))
    tf.get_variable_scope().set_initializer(tf.random_uniform_initializer(
          -FLAGS.init_scale, FLAGS.init_scale))
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope("seq2seq_gpu_%d" % i) as scope:
            input_batch = input_fn()
            # input_batch = batch_queue.dequeue()
            features, labels = self._format_input(input_batch)
            self.max_seq_len = tf.shape(features[0], name="max_seq_len_gpu_%d" % i)[-1]

            logits, context = self.inference(features)
            loss, one_hot_loss = self.loss(logits, labels)

            tf.summary.scalar('tower_loss_%d' % i, loss)
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            tf.get_variable_scope().reuse_variables()
            grads = optimizer.compute_gradients(loss)
            grads = self._clip_gradients(grads)
            
            tower_grads.append(grads)
    # grads = self._average_gradients(tower_grads)
    summaries.append(tf.summary.scalar('learning_rate', lr))

    grad_summaries = []
    vs = {}
    for g, v in grads:
      if g is not None:
        # grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        abs_g = tf.abs(g)
        abs_v = tf.abs(v)
        max_summary = tf.summary.scalar("{}/grad/abs_max_grad".format(v.name), tf.reduce_max(abs_g))
        mean_summary = tf.summary.scalar("{}/grad/abs_mean_grad".format(v.name), tf.reduce_mean(abs_g))
        max_val_summary = tf.summary.scalar("{}/val/abs_max_val".format(v.name), tf.reduce_max(abs_v))
        mean_val_summary = tf.summary.scalar("{}/val/abs_mean_val".format(v.name), tf.reduce_mean(abs_v))
        vs[v.name] = [tf.reduce_max(g), tf.reduce_mean(g), tf.reduce_max(abs_g), tf.reduce_mean(abs_g)]
        # grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
        grad_summaries.extend([max_val_summary, mean_val_summary, max_summary, mean_summary])
    summaries.extend(grad_summaries)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # for var in tf.trainable_variables():
    #   summaries.append(tf.summary.histogram(var.op.name, var))
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variables_averages_op)
    train_op = apply_gradient_op
    summary_op = tf.summary.merge(summaries)

    return train_op, summary_op, loss

  def build_generator(self, input_fn):
    inputs = input_fn()
    features, labels = self._format_input(inputs)
    outputs = self.inference(features, is_infer=True)
    return outputs

def create_model_params(params=None):
  default_params = {
    "num_examples_per_epoch_for_train": 2e6,
    "num_examples_per_epoch_for_dev": 2e3,
    "embedding.init_scale": 0.04,
    "embedding.share": False,
    "inference.beam_search.beam_width": 0,
    "inference.beam_search.length_penalty_weight": 0.0,
    "inference.beam_search.choose_successors_fn": "choose_top_k",
    "optimizer.clip_embed_gradients": 0.,
    "optimizer.lr_decay_type": "exponential_decay",
    "optimizer.lr_decay_rate": 0.5,
    "optimizer.lr_start_decay_at": 0,
    "optimizer.lr_stop_decay_at": 1e9,
    "optimizer.lr_min_learning_rate": 1e-12,
    "optimizer.lr_staircase": True,
    "optimizer.sync_replicas": 0,
    "optimizer.sync_replicas_to_aggregate": 0,
  }
  if params:
    default_params.update(params)
  return default_params


def train_model():
  print ("Loading data...")
  data_dir = os.path.abspath(os.path.join(os.path.curdir, "data_single"))
  src_dict_path = os.path.join(data_dir, "src.dict")
  trg_dict_path = os.path.join(data_dir, "trg.dict")
  src_dict, trg_dict = read_vocab(src_dict_path, trg_dict_path)
  params = create_model_params({
    "src_vocab_size": len(src_dict),
    "trg_vocab_size": len(trg_dict)
  })
  data_filename = "youdao_encn_tokens_50k"

  # num_examples_per_epoch_for_dev = params["num_examples_per_epoch_for_dev"]
  # num_dev_iter = int(math.ceil(float(num_examples_per_epoch_for_dev)/ FLAGS.valid_batch_size))

  train_data_generator = DataGenerator(data_dir, data_filename, FLAGS.batch_size,
    FLAGS.num_epochs, src_dict, trg_dict, FLAGS.max_seq_len)

  print ("Start building model...")
  with tf.Graph().as_default(), tf.device("/cpu:0"):
    tf.set_random_seed(1234)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    # session_conf.gpu_options.per_process_gpu_memory_fraction = 1.
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      seq2seq = Seq2SeqModel(params)

      timestamp = str(int(time.time()))
      out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
      print ("Writing to {}\n".format(out_dir))

      checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
      checkpoint_prefix = os.path.join(checkpoint_dir, "model")
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

      coord = tf.train.Coordinator()
      input_fn = train_data_generator.create_input_fn_new(is_training=True)
      train_op, summary_op, loss, et = seq2seq.build_model_multi_gpu(input_fn)
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

      # for v in tf.trainable_variables():
      #   print (v.name)

      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      sess.run(tf.global_variables_initializer()) # options=run_options, run_metadata=run_metadata)
      sess.run(tf.local_variables_initializer())

      train_summary_dir = os.path.join(out_dir, "summaries", "train")
      train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # Training loop. For each batch...
      print ("Start training...")
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
      current_step = 0
      try:
        while True:
          if coord and coord.should_stop():
            break

          start_time = time.time()
          _, summaries, train_loss, tet = sess.run([train_op, summary_op, loss, et])
          duration = time.time() - start_timexw

          assert not np.isnan(train_loss), 'Model diverged with loss = NaN'

          if current_step % FLAGS.show_train_stat_every == 0:
            num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus
            train_summary_writer.add_summary(summaries, current_step)
            format_str = ('%s: step %d, loss = %.2f, loss2 = %.2f ,(%.1f examples/sec; %.3f '
                    'sec/batch)')
            print (format_str % (datetime.now().isoformat(), current_step, train_loss, 0,
                 examples_per_sec, sec_per_batch))

          if current_step % 100 == 0:
              summaries = sess.run(summary_op)
              train_summary_writer.add_summary(summaries, current_step)

          if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
          current_step += 1

      except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
      except KeyboardInterrupt:
        print("Training interupted")
      except Exception as e:
        print("Something went wrong", e)
        raise
      finally:
        s = time.time()
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print ("saver time:", time.time() - s)
        print("Finally saved model checkpoint to {}\n".format(path))
        coord.request_stop()
        coord.join(threads)


def test_generator():
  data_dir = os.path.abspath(os.path.join(os.path.curdir, "data"))
  src_dict_path = os.path.join(data_dir, "src.dict")
  trg_dict_path = os.path.join(data_dir, "trg.dict")
  src_dict, trg_dict = read_vocab(src_dict_path, trg_dict_path)
  train_data_path = os.path.join(data_dir, "train/train")
  dev_data_path = os.path.join(data_dir, "test/test")
  g = DataGenerator(train_data_path, FLAGS.batch_size,
    FLAGS.num_epochs, src_dict, trg_dict, FLAGS.max_seq_len).build_data_generator()
  print (g.next())

if __name__ == '__main__':
  FLAGS._parse_flags()
  print("\nParameters:")
  for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
  print("")
  tf.logging.set_verbosity(tf.logging.ERROR)
  train_model()
  # test_generator()