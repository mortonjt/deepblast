import torch


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
