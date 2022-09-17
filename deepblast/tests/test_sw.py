import pytest
import torch
from deepblast.sw import wavefrontify, unwavefrontify
import deepblast.alignment_util as alignment


def random_sim_mat(b, l1, l2, emb_dim=3):
  seq_emb1 = torch.randn((b, l1, emb_dim))
  seq_emb2 = torch.randn((b, l2, emb_dim))
  return torch.einsum('nik,njk->nij', seq_emb1, seq_emb2)


def random_gap_penalty(minval, maxval, b=None, l1=None, l2=None):
  uniform = torch.distributions.uniform.Uniform(minval, maxval)
  if b is None:
      return torch.random.uniform((), minval=minval, maxval=maxval)
  elif l1 is None or l2 is None:
      return uniform.sample([b])
  else:
      return uniform.sample([b, l1, l2])


def best_alignment_brute_force(weights):
  len_1, len_2, _ = weights.shape
  best_alignment = None
  best_value = -np.inf
  for alignment_mat in npy_ops.alignment_matrices(len_1, len_2):
    value = np.vdot(alignment_mat, weights)
    if value > best_value:
      best_value = value
      best_alignment = alignment_mat
  return best_alignment


@pytest.mark.parametrize('b', [1, 8])
def test_wavefrontify(b):
    l1, l2, s = 14, 37, 9
    minval_open, maxval_open = 10.5, 11.5
    minval_extend, maxval_extend = 0.8, 1.2

    sim_mat = random_sim_mat(b, l1=l1, l2=l2, emb_dim=3)
    gap_open = random_gap_penalty(minval_open, maxval_open, b, l1, l2)
    gap_extend = random_gap_penalty(minval_extend, maxval_extend, b, l1, l2)
    w = alignment.weights_from_sim_mat(sim_mat, gap_open, gap_extend)

    w_wavefrontified = wavefrontify(w)
    w_unwavefrontified = unwavefrontify(w_wavefrontified)

    assert w_wavefrontified.shape == (l1 + l2 - 1, s, l1, b)
    assert torch.allclose(w_unwavefrontified, w)
    for n in torch.arange(b):
      for a in torch.arange(s):
        for i in torch.arange(l1):
          for j in torch.arange(l2):
            assert w_wavefrontified[i + j, a, i, n] == w[n, i, j, a]
