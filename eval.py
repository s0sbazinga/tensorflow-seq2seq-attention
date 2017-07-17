"""
From http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from seq2seq import DataGenerator, Seq2SeqModel, read_vocab, create_model_params

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('timestamp', None,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_dir', './runs/%s/summaries/dev' % FLAGS.timestamp,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'dev',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './runs/%s/checkpoints' % FLAGS.timestamp,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 2000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, dev_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.valid_batch_size))

      step = 0
      total_loss, total_len = 0., 0
      while step < num_iter and not coord.should_stop():
        loss, num_tokens = sess.run(dev_op)
        total_len += num_tokens
        total_loss += num_tokens * loss
        step += 1

      print('%s: dev loss = %.3f' % (datetime.now(), total_loss))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  print (FLAGS.checkpoint_dir)
  data_dir = os.path.abspath(os.path.join(os.path.curdir, "data"))
  src_dict_path = os.path.join(data_dir, "src.dict")
  trg_dict_path = os.path.join(data_dir, "trg.dict")
  src_dict, trg_dict = read_vocab(src_dict_path, trg_dict_path)
  params = create_model_params({
    "src_vocab_size": len(src_dict),
    "trg_vocab_size": len(trg_dict)
  })
  data_filename = "youdao_encn_tokens_50k"
  dev_data_generator = DataGenerator(data_dir, data_filename, FLAGS.valid_batch_size,
    1, src_dict, trg_dict, FLAGS.max_seq_len)

  with tf.Graph().as_default() as g:
    tf.set_random_seed(1234)

    seq2seq = Seq2SeqModel(params)
    
    input_fn = dev_data_generator.create_input_fn_new(is_training=False)
    _, loss, summary_op, num_tokens = seq2seq.build_eval_model(input_fn)
    dev_op = [loss, num_tokens]

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, dev_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
