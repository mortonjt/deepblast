import torch
import torch.nn as nn
import torch.nn.functional as F



def tt_slice3(x, a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return x[a1:a1+b1, a2:a2+b2, a3:a3+b3]

def tt_slice4(x, a, b):
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    return x[a1:a1+b1, a2:a2+b2, a3:a3+b3, a4:a4+b4]


def wavefrontify(tensor):
    """Rearranges input tensor for vectorized wavefront algorithm.

    Parameters
    ----------
    tensor : torch.Tensor :
        A torch.Tensor<dtype>[batch, len1, len2, s], where the first and last
        dimensions will be effectively treated as batch dimensions.

    Returns
    -------
    torch.Tensor
        A single torch.Tensor<dtype>[len1 + len2 - 1, s, len1, batch] satisfying
          out[k][a][i][n] = t[n][i][k - i][a]
        if the RHS is well-defined, and 0 otherwise.
        In other words, for each len1 x len2 matrix t[n, :, :, a], out[:, a, :, n]
        is a (len1 + len2 - 1) x len1 matrix whose rows correspond to antidiagonals
        of t[n, :, :, a].
    """
    b = tensor.shape[0]
    # l1, l2 = tf.shape(tensor)[1], tf.shape(tensor)[2]
    l1, l2 = tensor.shape[1], tensor.shape[2]
    s = tensor.shape[3]
    n_pad, padded_len = l1 - 1, l1 + l2 - 1

    ta = []
    for i in torch.arange(0, l1):
        row_i = tt_slice4(tensor, [0, i, 0, 0], [b, 1, l2, s]).squeeze(1)
        row_i = F.pad(row_i, [0, 0, n_pad, n_pad, 0, 0])
        #row_i = torch.nn.ConstantPad3d([[0, 0], [n_pad, n_pad], [0, 0]], 0)(row_i)
        row_i = tt_slice3(row_i, [0, n_pad - i, 0], [b, padded_len, s])
        ta.append(row_i)
    ta = torch.stack(ta, 0)

    return ta.permute(2, 3, 0, 1)  # out[padded_len, s, l1, b]


def unwavefrontify(tensor):
    """Inverts the "wavefrontify" transform.

    Parameters
    ----------
    torch.Tensor: A torch.Tensor<dtype>[len1 + len2 - 1, s, len1, batch], where the
       second and last dimensions will be effectively treated as batch dimensions.

    Returns
    -------
    A single tf.Tensor<dtype>[len1 + len2 - 1, s, len1, batch] satisfying
        out[n][i][j][a] = t[i + j][a][i][n].
    In other words, unwavefrontify(wavefrontify(t)) = t.
    """
    padded_len = tensor.shape[0]
    s, l1, b = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    l2 = padded_len - l1 + 1

    ta = []
    for i in torch.arange(0, l1):
        row_i = tt_slice4(tensor, [i, 0, i, 0], [l2, s, 1, b]).squeeze(2)
        ta.append(row_i)  # row_i[l2, s, b]
    ta = torch.stack(ta)  # ta[l1, l2, s, b]

    return ta.permute(3, 0, 1, 2)  # out[b, l1, l2, s]



## Auxiliary functions + syntatic sugar.
def slice_lead_dims(
        t,
        k,
        s,
):
    """Returns t[k][:s] for "wavefrontified" tensors."""
    return tt_slice4(t, [k, 0, 0, 0], [1, s, l1, b]).squeeze(0)

# "Wavefrontified" tensors contain invalid entries that need to be masked.
def slice_inv_mask(k):
    """Masks invalid and sentinel entries in wavefrontified tensors."""
    j_range = k - torch.range(1, l1 + 1).long() + 2
    return torch.logical_and(j_range > 0, j_range <= l2)  # True iff valid.

# Setups reduction operators.
def reduce_max_with_argmax(t, axis = 0):
    t_max = torch.max(t, axis=axis)
    t_argmax = torch.argmax(t, axis=axis).long()
    return t_max, t_argmax


def hard_sw_affine(
    weights,
    tol = 1e-6,
):
    """Solves the Smith-Waterman LP, computing both optimal scores and alignments.
    Args:
      weights: A tf.Tensor<float>[batch, len1, len2, 9] (len1 <= len2) of edge
        weights (see function alignment.weights_from_sim_mat for an in-depth
        description).
      tol: A small positive constant to ensure the first transition begins at the
        start state. Note(fllinares): this might not be needed anymore, test!
    Returns:
      Two tensors corresponding to the scores and alignments, respectively.
      + The first tf.Tensor<float>[batch] contains the Smith-Waterman scores for
        each pair of sequences in the batch.
      + The second tf.Tensor<int>[batch, len1, len2, 9] contains binary entries
        indicating the trajectory of the indices along the optimal path for each
        sequence pair, by having a one along the taken edges, with nine possible
        edges for each i,j.
    """
    # Gathers shape and type variables.
    b, l1, l2 = weights.shape[0], weights.shape[1], weights.shape[2]
    padded_len = l1 + l2 - 1
    dtype = weights.dtype
    inf = alignment.large_compatible_positive(dtype)

    # Rearranges input tensor for vectorized wavefront iterations.
    weights = wavefrontify(weights)  # [padded_len, s, l1, b]
    w_m, w_x, w_y = torch.split(weights, [4, 2, 3], axis=1)

    ### FORWARD

    # Initializes forward recursion.
    v_p2, v_p1 = torch.fill([3, l1, b], -inf), torch.fill([3, l1, b], -inf)
    # Ensures that edges cases for which all substitution costs are negative
    # result in a score of zero and an empty alignment.
    v_opt = torch.zeros(b, dtype=dtype)
    k_opt, i_opt = -torch.ones(b, dtype=torch.int32), -torch.ones(b, dtype=torch.int32)
    d_all = torch.zeros(padded_len)

    # Runs forward Smith-Waterman recursion.
    for k in range(padded_len):
        # NOTE(fllinares): shape information along the batch dimension seems to get
        # lost in the edge-case b=1
        # torch.autograph.experimental.set_loop_options(
        #     shape_invariants=[(v_p2, torch.TensorShape([3, None, None])),
        #                       (v_p1, torch.TensorShape([3, None, None])),
        #                       (v_opt, torch.TensorShape([None,])),
        #                       (k_opt, torch.TensorShape([None,])),
        #                       (i_opt, torch.TensorShape([None,]))])
        # inv_mask: masks out invalid entries for v_p2, v_p1 and v_opt updates.
        inv_mask_k = slice_inv_mask(k)[torch.newaxis, :, torch.newaxis]

        o_m = slice_lead_dims(w_m, k, 4) + alignment.top_pad(v_p2, tol)
        o_x = slice_lead_dims(w_x, k, 2) + v_p1[:2]
        v_p1 = alignment.left_pad(v_p1[:, :-1], -inf)
        o_y = slice_lead_dims(w_y, k, 3)  + v_p1

        v_m, d_m = reduce_max_with_argmax(o_m, axis=0)
        v_x, d_x = reduce_max_with_argmax(o_x, axis=0)
        v_y, d_y = reduce_max_with_argmax(o_y, axis=0)
        v = torch.where(inv_mask_k, torch.stack([v_m, v_x, v_y]), -inf)
        d = torch.stack([d_m, d_x + 1, d_y + 1])  # Accounts for start state (0).

        v_p2, v_p1 = v_p1, v
        v_opt_k, i_opt_k = reduce_max_with_argmax(v[0], axis=0)
        update_cond = v_opt_k > v_opt
        v_opt = torch.where(update_cond, v_opt_k, v_opt)
        k_opt = torch.where(update_cond, k, k_opt)
        i_opt = torch.where(update_cond, i_opt_k, i_opt)
        d_all = d_all.write(k, d)

    ### BACKTRACKING

    # Creates auxiliary tensors to encode backtracking "actions".
    steps_k = torch.tensor([0, -2, -1, -1]).long()
    steps_i = torch.tensor([0, -1, 0, -1]).long()
    trans_enc = torch.tensor([[10, 10, 10, 10],
                             [1, 2, 3, 4],
                             [10, 5, 6, 10],
                             [10, 7, 8, 9]]).long()  # [m_curr, m_prev]
    samp_idx = torch.arange(b)

    # Initializes additional backtracking variables.
    m_opt = torch.ones(b)  # Init at match states (by definition).
    paths_sp = torch.zeros(padded_len)
    # Runs Smith-Waterman backtracking.
    for k in range(padded_len - 1, -1, -1):
        # NOTE(fllinares): shape information along the batch dimension seems to get
        # lost in the edge-case b=1
        # torch.autograph.experimental.set_loop_options(
        #     shape_invariants=[(m_opt, torch.TensorShape([None,]))])
        # Computes tentative next indices for each alignment.
        k_opt_n = k_opt + torch.gather(steps_k, m_opt)
        i_opt_n = i_opt + torch.gather(steps_i, m_opt)
        # Computes tentative next state types for each alignment.
        m_opt_n_idx = torch.stack(
            [torch.maximum(m_opt - 1, 0), torch.maximum(i_opt, 0), samp_idx], -1)
        m_opt_n = torch.gather_nd(d_all.read(k), m_opt_n_idx)
        # Computes tentative next sparse updates for paths tensor.
        edges_n = torch.gather_nd(trans_enc, torch.stack([m_opt, m_opt_n], -1))
        paths_sp_n = torch.stack([samp_idx, i_opt + 1, k_opt - i_opt + 1, edges_n], -1)

        # Indicates alignments to be updated in this iteration.
        cond = torch.logical_and(k_opt == k, m_opt != 0)
        # Conditionally applies updates for each alignment.
        k_opt = torch.where(cond, k_opt_n, k_opt)
        i_opt = torch.where(cond, i_opt_n, i_opt)
        m_opt = torch.where(cond, m_opt_n, m_opt)
        paths_sp_k = torch.where(cond[:, None], paths_sp_n, torch.zeros([b, 4], torch.int32))
        paths_sp = paths_sp.write(k, paths_sp_k)  # [0, 0, 0, 0] used as dummy upd.

    # Applies sparse updates, building paths tensor.
    paths_sp = torch.reshape(paths_sp.stack(), [-1, 4])  # [(padded_len * b), 4]
    paths_sp_idx, paths_sp_upd = paths_sp[:, :3], paths_sp[:, 3]
    paths = torch.scatter_nd(paths_sp_idx, paths_sp_upd, [b, l1 + 1, l2 + 1])
    paths = paths[:, 1:, 1:]  # Removes sentinel row/col.
    # Represents paths tensor using one-hot encoding over 9 states.
    paths = torch.one_hot(paths, torch.reduce_max(trans_enc))[:, :, :, 1:]
    return v_opt, paths
