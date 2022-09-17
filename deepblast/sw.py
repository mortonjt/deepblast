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
    # Note(fllinares): I haven't yet managed to beat the performance of this
    # (wasteful) implementation with tf.argmax + tf.gather / tf.gather_nd :(
    t_max = torch.max(t, axis=axis)
    t_argmax = torch.argmax(t, axis=axis).long()
    return t_max, t_argmax
