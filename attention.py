"""Based on https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.layers.core import dense
from tensorflow.python.util import nest

import math
from collections import namedtuple
from functools import partial
from beam_search import create_initial_beam_state, beam_search_step

_concat = rnn_cell_impl._concat
_like_rnncell = rnn_cell_impl._like_rnncell
_linear = rnn_cell_impl._linear
_rnn_step = rnn._rnn_step
_infer_state_dtype = rnn._infer_state_dtype
_transpose_batch_time = rnn._transpose_batch_time


class AttentionDecoder(object):
  ExtraOutput = namedtuple("ExtraOutput", ["logits", "predicted_ids", "beam_parent_ids"])
  def __init__(self, embeddings, logit_fn, is_infer=False, infer_params=None):
    self.embeddings = embeddings
    self.logit_fn = logit_fn
    self.is_infer = is_infer
    self.vocab_size = embeddings.get_shape().as_list()[0]
    if is_infer:
      self.infer_params = infer_params
      self.initial_search_state = create_initial_beam_state(infer_params["beam_width"])

  @property
  def extra_dtype(self):
    return AttentionDecoder.ExtraOutput(logits=dtypes.float32,
                                        predicted_ids=dtypes.int32,
                                        beam_parent_ids=dtypes.int32)

  def attention_step(self, state, state_size, context, att_sequence_length=None, last_coverage=None,
                     encoded_fertility=None, use_coverage=True):
    def _attention(state, context):
        with vs.variable_scope("attention"):
          ctx_shape = context.get_shape().as_list()
          dim_ctx = ctx_shape[-1]
          if isinstance(state_size, tuple):
              _, m_prev = state
              _, m_size = state_size
          else:
              m_prev, m_size = state, state_size
          with vs.variable_scope("ctx_proj"):
            pctx = dense(context, units=dim_ctx, use_bias=True)

          with vs.variable_scope("state_proj"):
            pstate = array_ops.expand_dims(_linear(m_prev, dim_ctx, bias=False), axis=1)

          with vs.variable_scope("cell_proj") as cell_proj_scope:
            alpha = math_ops.reduce_sum(math_ops.tanh(pstate + pctx), [2])

          if att_sequence_length is not None:
            alpha_mask = array_ops.sequence_mask(lengths=att_sequence_length,
              maxlen=ctx_shape[1], dtype=dtypes.float32)
            alpha = alpha * alpha_mask + ((1.0 - alpha_mask) * dtypes.float32.min)
          alpha_normalized = nn_ops.softmax(alpha)
          ctx = math_ops.reduce_sum(context * array_ops.expand_dims(alpha_normalized, axis=2), axis=1)

          return ctx, alpha_normalized

    def _attention_with_coverage(state, context, last_coverage, encoded_fertility):
        with vs.variable_scope("attention"):
          ctx_shape = context.get_shape().as_list()
          dim_ctx = ctx_shape[-1]
          if isinstance(state_size, tuple):
              _, m_prev = state
              _, m_size = state_size
          else:
              m_prev, m_size = state, state_size
          # print (last_coverage.get_shape().as_list())

          init_std = 1. / math.sqrt(m_size)
          cov_initializer = init_ops.random_normal_initializer(mean=0, stddev=1.)
          initializer = init_ops.random_normal_initializer(mean=0, stddev=init_std)
          with vs.variable_scope("ctx_proj"):
            pcoverage = dense(array_ops.expand_dims(last_coverage, -1), units=dim_ctx,
                              kernel_initializer=cov_initializer, use_bias=False)
            pctx = dense(context, units=dim_ctx, 
                         kernel_initializer=initializer, use_bias=True)
            # pctx = _linear(array_ops.reshape(pctx, [-1, dim_ctx + 1]), dim_ctx, bias=True)
            # pctx = array_ops.reshape(pctx, [-1, ctx_shape[1], dim_ctx])
            # pctx = array_ops.reshape(context, [-1, dim_ctx])
            # pctx = array_ops.reshape(_linear(pctx, dim_ctx, bias=True), [-1, ctx_shape[1], dim_ctx])

          with vs.variable_scope("state_proj"):
            pstate = array_ops.expand_dims(_linear(m_prev, dim_ctx, kernel_initializer=initializer, bias=False), axis=1)

          with vs.variable_scope("cell_proj") as cell_proj_scope:
            # alpha = math_ops.reduce_sum(math_ops.tanh(pstate + pctx + pcoverage), [2])
            alpha = dense(math_ops.tanh(pstate + pctx + pcoverage), units=1,
                          kernel_initializer=initializer, use_bias=False)
            alpha = math_ops.reduce_sum(alpha, [2])
            # pctx = math_ops.tanh(array_ops.reshape((pctx + pstate), [-1, dim_ctx]))
            # alpha = array_ops.reshape(_linear(pctx, 1, bias=True), [-1, ctx_shape[1]])

          if att_sequence_length is not None:
            alpha_mask = array_ops.sequence_mask(lengths=att_sequence_length,
              maxlen=ctx_shape[1], dtype=dtypes.float32)
            alpha = alpha * alpha_mask + ((1.0 - alpha_mask) * dtypes.float32.min)
          alpha_normalized = nn_ops.softmax(alpha)
          ctx = math_ops.reduce_sum(context * array_ops.expand_dims(alpha_normalized, axis=2), axis=1)
          # print (alpha_normalized, last_coverage, encoded_fertility)

          encoded_fertility = array_ops.identity(encoded_fertility, name="encoded_fertility")
          new_coverage = last_coverage + alpha_normalized * math_ops.pow(2 * encoded_fertility, -1)
          new_coverage = new_coverage * alpha_mask + ((1.0 - alpha_mask) * last_coverage)
          return ctx, alpha_normalized, new_coverage

    if use_coverage:
      assert (last_coverage is not None and encoded_fertility is not None), "No coverage input."
      return _attention_with_coverage(state, context, last_coverage, encoded_fertility)
    else:
      return _attention(state, context)

  def search_step(self, time, decoder_output, logits, state):
    decoder_state, beam_state = state

    bs_output, beam_state = beam_search_step(time_=time, logits=logits,
        beam_state=beam_state, params=self.infer_params)

    decoder_state = nest.map_structure(
        lambda x: array_ops.gather(x, bs_output.beam_parent_ids), decoder_state)
    decoder_output = nest.map_structure(
        lambda x: array_ops.gather(x, bs_output.beam_parent_ids), decoder_output)

    next_state = (decoder_state, beam_state)

    predicted_ids = bs_output.predicted_ids
    beam_parent_ids = bs_output.beam_parent_ids
    search_finished = math_ops.equal(predicted_ids, self.infer_params["eos_token"])

    new_input = embedding_ops.embedding_lookup(self.embeddings, predicted_ids)
    extra_output = AttentionDecoder.ExtraOutput(logits=logits, predicted_ids=predicted_ids,
                                                beam_parent_ids=beam_parent_ids)

    return (decoder_output, extra_output, (decoder_state, beam_state),
            new_input, search_finished)


def dynamic_attention_rnn(decoder, cell, inputs, context, sequence_length=None, att_sequence_length=None,
                          use_coverage=False, initial_state=None, dtype=None, parallel_iterations=None,
                          swap_memory=False, time_major=False, scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.
  Performs fully dynamic unrolling of `inputs`.
  `Inputs` may be a single `Tensor` where the maximum time is either the first
  or second dimension (see the parameter
  `time_major`).  Alternatively, it may be a (possibly nested) tuple of
  Tensors, each of them having matching batch and time dimensions.
  The corresponding output is either a single `Tensor` having the same number
  of time steps and batch size, or a (possibly nested) tuple of such tensors,
  matching the nested structure of `cell.output_size`.
  The parameter `sequence_length` is optional and is used to copy-through state
  and zero-out outputs when past a batch element's sequence length. So it's more
  for correctness than performance.
  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".
  Returns:
    A pair (outputs, state) where:
      outputs: The RNN output `Tensor`.
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
        Note, if `cell.output_size` is a (possibly nested) tuple of integers
        or `TensorShape` objects, then `outputs` will be a tuple having the
        same structure as `cell.output_size`, containing Tensors having shapes
        corresponding to the shape data in `cell.output_size`.
      state: The final state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes.
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not _like_rnncell(cell):
    raise TypeError("cell must be an instance of RNNCell")

  # By default, time_major==False and inputs are batch-major: shaped
  #   [batch, time, depth]
  # For internal calculations, we transpose to [time, batch, depth]
  flat_input = nest.flatten(inputs)

  if not time_major:
    # (B,T,D) => (T,B,D)
    flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
    flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

  parallel_iterations = parallel_iterations or 32
  if sequence_length is not None:
    sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length.get_shape().ndims not in (None, 1):
      raise ValueError(
          "sequence_length must be a vector of length batch_size, "
          "but saw shape: %s" % sequence_length.get_shape())
    sequence_length = array_ops.identity(  # Just to find it in the graph.
        sequence_length, name="sequence_length")

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "att_rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)
    input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
    batch_size = input_shape[0][1]

    for input_ in input_shape:
      if input_[1].get_shape() != batch_size.get_shape():
        raise ValueError("All inputs should have the same batch size")

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, extra, final_state) = _dynamic_att_loop(
        decoder,
        cell,
        inputs,
        state,
        context,
        use_coverage=use_coverage,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        att_sequence_length=att_sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = nest.map_structure(_transpose_batch_time, outputs)
      final_extra = nest.map_structure(_transpose_batch_time, extra)

    return (outputs, final_extra, final_state)


def _dynamic_att_loop(decoder,
                      cell,
                      inputs,
                      initial_state,
                      context,
                      parallel_iterations,
                      swap_memory,
                      use_coverage=False,
                      sequence_length=None,
                      att_sequence_length=None,
                      dtype=None):
  """Internal implementation of Dynamic RNN.
  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
      tuple of such elements.
    initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
      `cell.state_size` is a tuple, then this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    parallel_iterations: Positive Python int.
    swap_memory: A Python boolean
    sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
    dtype: (optional) Expected dtype of output. If not specified, inferred from
      initial_state.
  Returns:
    Tuple `(final_outputs, final_state)`.
    final_outputs:
      A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
      `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
      objects, then this returns a (possibly nsted) tuple of Tensors matching
      the corresponding shapes.
    final_state:
      A `Tensor`, or possibly nested tuple of Tensors, matching in length
      and shapes to `initial_state`.
  Raises:
    ValueError: If the input depth cannot be inferred via shape inference
      from the inputs.
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = input_shape[1]

  is_infer = decoder.is_infer
  if use_coverage:
    encoded_fert_init, encoded_fertility, context = context
    encoded_fert_init = array_ops.identity(encoded_fert_init, name="encoded_fert_init")
  else:
    encoded_fert_init, encoded_fertility = 0, 0
    # coverage_shape = encoded_fert_init.get_shape().as_list()
    # flat_coverage = nest.flatten(encoded_fert_init)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_att_rnn") as scope:
    base_name = scope

  def _create_ta(steps, name, dtype):
    if not is_infer:
      return tensor_array_ops.TensorArray(dtype=dtype,
                                          size=steps,
                                          tensor_array_name=base_name + name)
    else:
      return tensor_array_ops.TensorArray(dtype=dtype,
                                          size=0,
                                          dynamic_size=True,
                                          tensor_array_name=base_name + name)

  output_ta = tuple(_create_ta(time_steps, "output_%d" % i,
                               _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
  if is_infer:
    create_extra_ = lambda s, n: nest.map_structure(partial(_create_ta, s, n),
                                                    decoder.extra_dtype)
    extra_output_ta = tuple(create_extra_(time_steps, "extra_output_%d" % i)
                      for i in range(len(flat_output_size)))
  else:
    extra_output_ta = tuple(_create_ta(time_steps, "logits_%d" % i,
                                       _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))

  input_ta = tuple(_create_ta(time_steps, "input_%d" % i, flat_input[i].dtype)
                   for i in range(len(flat_input)))

  input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

  # coverage_ta = tuple(_create_ta(time_steps + 1, "coverage_%d" % i,
  #                                _infer_state_dtype(dtype, state))
  #                   for i in range(len(flat_coverage)))
  # if use_coverage:
  #   coverage_ta = tuple(ta.write(0, coverage_)
  #                       for ta, coverage_ in zip(coverage_ta, flat_coverage))

  def _time_step(time, input_ta_t, output_ta_t, extra_output_ta_t,
                 last_coverage, state, finished):
    """Take a time step of the dynamic RNN.
    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.
    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    input_t = tuple(ta.read(time) for ta in input_ta_t)
    # Restore some shape information
    for input_, shape in zip(input_t, inputs_got_shape):
      input_.set_shape(shape[1:])

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)

    rnn_state = state if not is_infer else state[0]
    if use_coverage:
      # coverage_t = tuple(ta.read(time) for ta in coverage_ta_t)
      # coverage_t = nest.pack_sequence_as(structure=encoded_fert_init, flat_sequence=coverage_t)
      # coverage_t.set_shape(coverage_shape)
      ctx, att_weights, new_coverage = decoder.attention_step(rnn_state, state_size, context,
                                          att_sequence_length, last_coverage, encoded_fertility)

      # new_coverage = nest.flatten(new_coverage)
      # coverage_ta_t = tuple(ta.write(time + 1, coverage)
      #                       for ta, coverage in zip(coverage_ta_t, new_coverage))
    else:
      new_coverage = last_coverage
      ctx, att_weights = decoder.attention_step(rnn_state, state_size, context, use_coverage=False)

    call_cell = lambda: cell(array_ops.concat([input_t, ctx], 1), rnn_state)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=rnn_state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()
      assert is_infer, "Manually zero output when inferring."
      output = nest.map_structure(
        lambda out, zero: array_ops.where(finished, zero, out),
        output,
        zero_output)

    logits = decoder.logit_fn(output)
    if is_infer:
      mix_state = (new_state, state[1])
      (new_output, extra_output, new_state,
       new_input_t, search_finished) = decoder.search_step(time, output, logits, mix_state)

      new_input_t = nest.flatten(new_input_t)
      input_ta_t = tuple(
        ta.write(time + 1, new_input_) for ta, new_input_ in zip(input_ta_t, new_input_t))

      new_finished = math_ops.logical_or(search_finished, finished)
      extra_output = nest.pack_sequence_as(structure=extra_output_ta, flat_sequence=extra_output)
      output = new_output
    else:
      extra_output = logits
      extra_output = nest.flatten(extra_output)
      new_finished = finished

    # Pack state if using state tuples
    output = nest.flatten(output)

    output_ta_t = tuple(
        ta.write(time, out) for ta, out in zip(output_ta_t, output))

    if is_infer:
      write_ta_ = lambda t, o: t.write(time, o)
      map_write_ = lambda ta, out: nest.map_structure(write_ta_, ta, out)
      extra_output_ta_t = tuple(
          map_write_(ta, out) for ta, out in zip(extra_output_ta_t, extra_output))
    else:
      extra_output_ta_t = tuple(
          ta.write(time, out) for ta, out in zip(extra_output_ta_t, extra_output))

    return (time + 1, input_ta_t, output_ta_t, extra_output_ta_t,
            new_coverage, new_state, new_finished)

  if is_infer:
    init_finished = array_ops.tile([False], [batch_size])
    init_state = (state, decoder.initial_search_state)
    condition_fn = lambda *args: math_ops.logical_not(math_ops.reduce_all(args[-1]))
  else:
    init_finished = False
    init_state = state
    condition_fn = lambda time, *_: time < time_steps

  res = control_flow_ops.while_loop(
      cond=condition_fn,
      body=_time_step,
      loop_vars=(time, input_ta, output_ta, extra_output_ta,
                 encoded_fert_init, init_state, init_finished),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  final_outputs = tuple(ta.stack() for ta in res[2])
  final_extra = tuple(nest.map_structure(lambda ta: ta.stack(), ta)
                      for ta in res[3])
  final_state = res[-2]

  # Restore some shape information
  if not is_infer:
    vocab_size = decoder.vocab_size
    for output, extra, output_size in zip(final_outputs, final_extra, flat_output_size):
      shape = _concat(
          [const_time_steps, const_batch_size], output_size, static=True)
      output.set_shape(shape)
      extra_shape = _concat(
          [const_time_steps, const_batch_size], vocab_size, static=True)
      extra.set_shape(extra_shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    final_extra = nest.pack_sequence_as(
        structure=vocab_size, flat_sequence=final_extra)

  return (final_outputs, final_extra, final_state)


class LSTMAttCell(rnn_cell_impl.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, context, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:
      if self._num_unit_shards is not None:
        unit_scope.set_partitioner(
            partitioned_variables.fixed_size_partitioner(
                self._num_unit_shards))
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)
      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope) as projection_scope:
          if self._num_unit_shards is not None:
            projection_scope.set_partitioner(None)
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
             self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection") as proj_scope:
          if self._num_proj_shards is not None:
            proj_scope.set_partitioner(
                partitioned_variables.fixed_size_partitioner(
                    self._num_proj_shards))
          m = _linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state
