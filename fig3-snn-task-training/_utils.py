# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import os.path
from typing import Any, Optional, Mapping, Callable, Union

import brainpy_datasets as bd
import brainstate as bst
import brainunit as u
import jax
import numpy as np
import orbax.checkpoint
from jax.experimental.sparse.csr import csr_matvec_p, csr_matmat_p
from torch.utils.data import DataLoader, IterableDataset

__all__ = ['Checkpointer', 'TaskLoader', 'CSRLayer']


class Checkpointer(orbax.checkpoint.CheckpointManager):
  def __init__(
      self,
      directory: str,
      max_to_keep: Optional[int] = None,
      save_interval_steps: int = 1,
      metadata: Optional[Mapping[str, Any]] = None,
  ):
    options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      save_interval_steps=save_interval_steps,
      create=True
    )
    super().__init__(os.path.abspath(directory), options=options, metadata=metadata)

  def save(self, args: Any, step: int, metrics: Optional[Any] = None, force: Optional[bool] = False, **kwargs):
    r = super().save(step, metrics=metrics, force=force, args=orbax.checkpoint.args.StandardSave(args))
    self.wait_until_finished()
    return r

  def restore(self, args: Any = None, step: int = None, items: Any = None, **kwargs):
    self.wait_until_finished()
    step = self.latest_step() if step is None else step
    if args is not None:
      tree = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, args)
      args = orbax.checkpoint.args.StandardRestore(tree)
    return super().restore(step, items=items, args=args)


class TaskData(IterableDataset):
  def __init__(self, task: bd.cognitive.CognitiveTask):
    self.task = task

  def __iter__(self):
    while True:
      yield self.task.sample_a_trial(0)[:2]


class TaskLoader(DataLoader):
  def __init__(self, dataset: bd.cognitive.CognitiveTask, *args, **kwargs):
    assert isinstance(dataset, bd.cognitive.CognitiveTask)
    super().__init__(TaskData(dataset), *args, **kwargs)


def csr_matvec(
    data: bst.typing.ArrayLike,
    indices: bst.typing.ArrayLike,
    indptr: bst.typing.ArrayLike,
    v: bst.typing.ArrayLike,
    *,
    shape,
    transpose=False
) -> jax.Array:
  """
  Product of CSR sparse matrix and a dense vector.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    v : array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
  """
  return csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)


def csr_matmat(
    data: bst.typing.ArrayLike,
    indices: bst.typing.ArrayLike,
    indptr: bst.typing.ArrayLike,
    B: bst.typing.ArrayLike,
    *,
    shape,
    transpose: bool = False
) -> jax.Array:
  """Product of CSR sparse matrix and a dense matrix.

  Args:
    data : array of shape ``(nse,)``.
    indices : array of shape ``(nse,)``
    indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
    B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
    shape : length-2 tuple representing the matrix shape
    transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
    C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix-matrix product.
  """
  return csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)


class CSRLayer(bst.nn.DnnLayer):
  def __init__(
      self,
      n_pre: int,
      n_post: int,
      prob: float,
      w_init: Union[Callable, bst.typing.ArrayLike] = bst.init.KaimingNormal(scale=2 * u.siemens),
      w_sign: Optional[bst.typing.ArrayLike] = None,
      mode: bst.environ.Mode = None
  ):
    super().__init__(mode=mode)

    # parameters
    self.w_sign = w_sign
    self.n_pre = n_pre
    self.n_post = n_post
    self.prob = prob

    # connection
    n_conn = int(n_post * prob)
    self.indptr = np.arange(0, n_pre + 1) * n_conn
    self.indices = np.random.randint(0, n_post, (n_pre, n_conn)).flatten()

    # weight
    weight = bst.init.param(w_init, (self.n_pre, n_conn), allow_scalar=True)
    weight = u.math.flatten(weight)
    if self.mode.has(bst.mixin.Training):
      weight = bst.ParamState(weight)
    self.weight = weight

  # def to_dense_conn(self):
  #   data = self.weight.value if isinstance(self.weight, bst.State) else self.weight
  #   return bp.math.sparse.csr_to_dense(data, self.indices, self.indptr, shape=(self.conn.pre_num, self.conn.post_num))

  def update(self, x):
    weight = self.weight.value if isinstance(self.weight, bst.State) else self.weight
    unit = u.get_unit(x) * u.get_unit(weight)
    weight = u.Quantity(weight).mantissa
    x = u.Quantity(x).mantissa
    r = self._op(x, weight)
    return u.maybe_decimal(u.Quantity(r, unit=unit))

  def _op(self, x, w):
    if self.w_sign is None:
      w = u.math.abs(w)
    else:
      w = u.math.abs(w) * self.w_sign
    assert u.is_unitless(self.indices), 'indices should be unitless'
    assert u.is_unitless(self.indptr), 'indptr should be unitless'
    unit = u.get_unit(w) * u.get_unit(x)
    w = u.get_mantissa(w)
    x = u.get_mantissa(x)
    if x.ndim == 1:
      r = csr_matvec(
        w,
        self.indices,
        self.indptr,
        x,
        shape=(self.n_pre, self.n_post),
        transpose=True
      )
    else:
      shapes = x.shape[:-1]
      x = u.math.flatten(x, end_axis=-2)
      y = csr_matmat(
        w,
        self.indices,
        self.indptr,
        x.T,
        shape=(self.n_pre, self.n_post),
        transpose=True
      )
      y = y.T
      r = u.math.reshape(y, shapes + (y.shape[-1],))
    return u.maybe_decimal(u.Quantity(r, unit=unit))
