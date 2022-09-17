import torch
import torch.nn.functional as F


def top_pad(t, v):
  """Pads torch.Tensor `t` by prepending `v` along the leading dimension."""
  return F.pad(t, [1, 0, 0, 0, 0, 0], value=v)


def left_pad(t, v):
  """Pads torch.Tensor `t` by prepending `v` along the second leading dimension."""
  return F.pad(t, [0, 0, 1, 0, 0, 0], value=v)


def right_pad(t, v):
  """Pads torch.Tensor `t` by appending `v` along the second leading dimension."""
  return F.pad(t, [0, 0, 0, 1, 0, 0], value=v)


def weights_from_sim_mat(
        sim_mat,
        gap_open,
        gap_extend,
):
    """Computes the edge weights for the Smith-Waterman LP.
    Args:
    sim_mat: a torch.Tensor<float>[batch, len1, len2] with the substitution values
      for pairs of sequences.
    gap_open: a torch.Tensor<float>[batch, len1, len2] or torch.Tensor<float>[batch]
      of penalties for opening a gap.
    gap_extend: a torch.Tensor<float>[batch, len1, len2] or torch.Tensor<float>[batch]
      of penalties for extending a gap.
    Returns:
    A single torch.Tensor<float>[batch, len1, len2, 9] of edge weights for nine
    edge types. These correspond to a (strict) subset of allowed (from, to)
    state transitions between four state types, namely, start, match, gap_in_x
    and gap_in_y. Along the last dimension:
    + The first four (0:4) indices form a torch.Tensor<float>[batch, len1, len2, 4]
      of weights for all edges leading into match states. That is, these
      represent transitions (start, match), (match, match), (gap_in_x, match)
      and (gap_in_y, match), respectively.
    + The next two (4:6) indices form a torch.Tensor<float>[batch, len1, len2, 2]
      of weights for all edges leading into gap_in_x states. These represent
      transitions (match, gap_in_x) and (gap_in_x, gap_in_x), respectively. Note
      that, by convention, (gap_in_y, gap_in_x) transitions are disallowed.
    + The last three (6:9) indices form a torch.Tensor<float>[batch, len1, len2, 3]
      of weights for all edges leading into gap_in_y states. These represent
      transitions (match, gap_in_y) and (gap_in_x, gap_in_y) and, finally,
      (gap_in_y, gap_in_y), respectively.
    """
    l1, l2 = sim_mat.shape[1:3]

    sim_mat = sim_mat[Ellipsis, None]
    sim_mat = torch.tile(sim_mat, [1, 1, 1, 4])
    if len(gap_open.shape) == 3:
        gap_open = gap_open[Ellipsis, None]
        gap_extend = gap_extend[Ellipsis, None]
    else:
        gap_open = gap_open[Ellipsis, None, None, None]
        gap_open = torch.tile(gap_open, [1, l1, l2, 1])
        gap_extend = gap_extend[Ellipsis, None, None, None]
        gap_extend = torch.tile(gap_extend, [1, l1, l2, 1])

    weights_m = sim_mat
    weights_x = torch.concat([-gap_open, -gap_extend], axis=-1)
    weights_y = torch.concat([-gap_open, weights_x], axis=-1)

    return torch.concat([weights_m, weights_x, weights_y], axis=-1)


def adjoint_weights_from_sim_mat(
        weights,
        gap_open_shape,
        gap_extend_shape,
):
    """Computes the adjoint of `weights_from_sim_mat`.
    Viewing `weights_from_sim_mat` as a linear map weights = A sw_params, this
    function implements the linear map A^{T} weights. Primarily to be used when
    implementing custom_gradients in functions downstream.
    Args:
    weights: a tf.Tensor<float>[batch, len1, len2, 9].
    gap_open_shape: a tf.TensorShape representing the shape of gap_open in
      sw_params.
    gap_extend_shape: a tf.TensorShape representing the shape of gap_extend in
      sw_params.
    Returns:
    A tuple (sim_mat_out, gap_open_out, gap_extend_out) such that
      + sim_mat_out is a tf.Tensor<float>[batch, len1, len2] representing the
        elements of A^{T} weights corresponding to sim_mat.
      + gap_open_out is a tf.Tensor<float>[gap_open_shape] representing the
        elements of A^{T} weights corresponding to gap_open_shape.
      + gap_extend_out is a tf.Tensor<float>[gap_extend_shape] representing the
        elements of A^{T} weights corresponding to gap_extend_out.
  """
    sim_mat_out = torch.reduce_sum(weights[Ellipsis, :4], axis=-1)

    # Aggregates output across positions / examples too when appropriate.
    gap_open_out = - (weights[Ellipsis, 4] + weights[Ellipsis, 6] + weights[Ellipsis, 7])
    if len(gap_open_shape) == 1:
      gap_open_out = torch.sum(gap_open_out, axis=[1, 2])
    elif len(gap_open_shape) == 0:
      gap_open_out = torch.sum(gap_open_out)

    gap_extend_out = - (weights[Ellipsis, 5] + weights[Ellipsis, 8])
    if len(gap_extend_shape) == 1:
      gap_extend_out = torch.reduce_sum(gap_extend_out, axis=[1, 2])
    elif len(gap_extend_shape) == 0:
      gap_extend_out = torch.sum(gap_extend_out)

    return sim_mat_out, gap_open_out, gap_extend_out


def length(alignments_or_paths):
    """Computes the lengths in batch of sparse / dense alignments."""
    if len(alignments_or_paths.shape) == 3:  # Sparse format.
        pos_x, pos_y = alignments_or_paths[:, 0], alignments_or_paths[:, 1]
        padding_mask = torch.logical_and(pos_x > 0, pos_y > 0)
        return tf.sum(tf.cast(padding_mask, tf.float32), axis=-1)
    else:  # Dense format.
        return tf.sum(alignments_or_paths, axis=[1, 2, 3])


def state_count(alignments_or_paths, states):
    """Counts match/gap_open/gap_extend in batch of sparse / dense alignments."""
    if len(alignments_or_paths.shape) == 3:  # Sparse format.
        batch_size = alignments_or_paths.shape[0]
        state_indices = alignments_to_state_indices(alignments_or_paths, states)
        batch_indicators = state_indices[:, 0]
        ones = torch.ones_like(batch_indicators, tf.float32)
        # tf.math.unsorted_segment_sum(ones, batch_indicators, batch_size)
        return (torch.zeros(batch_indicators.shape[1], batch_size.shape[1])
                .scatter_add(1,  batch_indicators, batch_size))

    else:  # Dense format.
        state_indicators = paths_to_state_indicators(alignments_or_paths, states)
        return torch.sum(state_indicators, axis=[1, 2])


def endpoints(alignments_or_paths, start = True):
    """Computes the endpoints in batch of sparse / dense alignments."""
    if len(alignments_or_paths.shape) == 3:  # Sparse format.
        pos = alignments_or_paths[:, :2]
        return pos[Ellipsis, 0] if start else tf.reduce_max(pos, axis=-1)
    else:  # Dense format.
      shape = torch.shape(alignments_or_paths)
      batch_size = shape[0]
      len_x, len_y = shape[1], shape[2]
      matches = paths_to_state_indicators(alignments_or_paths, 'match')
      matches = torch.reshape(matches, [batch_size, -1])
      matches = matches if start else matches[:, ::-1]
      raveled_indices = torch.argmax(matches, axis=-1).long()
      start_x = torch.floor(raveled_indices / len_x).long()
      start_y = raveled_indices - start_x * len_x
      # Uses one-based indexing for consistency with sparse format.
      endpoint_x = start_x + 1 if start else len_x - start_x
      endpoint_y = start_y + 1 if start else len_y - start_y
      return torch.stack([endpoint_x, endpoint_y])


def path_label_squeeze(paths):
    """Returns a weights sum of paths solutions, for visualization."""
    v_range = torch.range(1, tf.shape(paths)[-1] + 1, dtype=paths.dtype)
    return torch.einsum('ijkn,n->ijk', paths, v_range)
