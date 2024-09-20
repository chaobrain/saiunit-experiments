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

import numbers
import os
import os.path
import platform
import time
from typing import Optional, Dict, Callable, Union, Sequence
from typing import Tuple

import matplotlib

from _arg_utls import MyArgumentParser

if platform.system().startswith('Linux'):
  matplotlib.use('agg')

parser = MyArgumentParser()

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
parser.add_argument("--opt", type=str, default='adam', help="Number of training epochs.")

# dataset
parser.add_argument("--batch_size", type=int, default=128, help="")
parser.add_argument("--warmup", type=float, default=0., help="The ratio for network simulation.")
parser.add_argument("--num_workers", type=int, default=4, help="")

# model
parser.add_argument("--dt", type=float, default=1., help="")
parser.add_argument("--neuron", type=str, default='gif', choices=['gif', 'alif'], help="")
parser.add_argument("--n_rec", type=int, default=200, help="")
parser.add_argument("--w_ei_ratio", type=float, default=4., help="")
parser.add_argument("--ff_scale", type=float, default=1., help="")
parser.add_argument("--rec_scale", type=float, default=0.5, help="")
parser.add_argument("--beta", type=float, default=1.0, help="")
parser.add_argument("--tau_a", type=float, default=1000., help="")
parser.add_argument("--tau_neu", type=float, default=100., help="")
parser.add_argument("--tau_syn", type=float, default=10., help="")
parser.add_argument("--tau_out", type=float, default=10., help="")
parser.add_argument("--conn_method", type=str, default='dense', help="")

# training parameters
parser.add_argument("--mode", type=str, default='train', help="")

# regularization parameters
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")
gargs = parser.parse_args()

import brainunit as u
import brainstate as bst
import brainpy as bp
import brainpy_datasets as bd
import braintools as bts
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from _utils import TaskLoader, CSRLayer

bst.environ.set(
  dt=gargs.dt * u.ms,
  mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching())
)


class SignedWLinear(bst.nn.DnnLayer):
  """
  Linear layer with signed weights.
  """

  def __init__(
      self,
      in_size: Union[int, Sequence[int]],
      out_size: Union[int, Sequence[int]],
      w_init: Union[Callable, bst.typing.ArrayLike] = bst.init.KaimingNormal(scale=2 * u.siemens),
      w_sign: Optional[bst.typing.ArrayLike] = None,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None
  ):
    super().__init__(name=name, mode=mode)

    # input and output shape
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)

    # w_mask
    self.w_sign = w_sign

    # weights
    weight = bst.init.param(w_init, self.in_size + self.out_size, allow_none=False)
    if self.mode.has(bst.mixin.Training):
      weight = bst.ParamState(weight)
    self.weight = weight

  def update(self, x):
    if self.mode.has(bst.mixin.Training):
      weight = self.weight.value
    else:
      weight = self.weight
    if self.w_sign is None:
      return u.math.matmul(x, u.math.abs(weight))
    else:
      return u.math.matmul(x, u.math.abs(weight) * self.w_sign)


class Expon(bst.nn.Synapse):
  def __init__(
      self,
      size: bst.typing.Size,
      keep_size: bool = False,
      name: Optional[str] = None,
      mode: Optional[bst.mixin.Mode] = None,
      tau: bst.typing.ArrayLike = 8.0 * u.ms,
  ):
    super().__init__(name=name, mode=mode, size=size, keep_size=keep_size)
    self.tau = bst.init.param(tau, self.varshape)

  def init_state(self, batch_size: int = None, **kwargs):
    self.g = bst.ShortTermState(bst.init.param(bst.init.Constant(0. * u.siemens), self.varshape, batch_size))

  def reset_state(self, batch_size: int = None, **kwargs):
    self.g.value = bst.init.param(bst.init.Constant(0. * u.siemens), self.varshape, batch_size)

  def update(self):
    self.g.value = self.g.value - self.g.value / self.tau * bst.environ.get_dt()
    return self.g.value

  def align_post_input_add(self, x):
    self.g.value = self.g.value + x


class GIF(bst.nn.Neuron):
  def __init__(
      self,
      size,
      V_rest: Union[bst.typing.ArrayLike, Callable] = 0. * u.mV,
      V_th_inf: Union[bst.typing.ArrayLike, Callable] = 1. * u.mV,
      R: Union[bst.typing.ArrayLike, Callable] = 1. * u.ohm,
      tau: Union[bst.typing.ArrayLike, Callable] = 20. * u.ms,
      tau_a: Union[bst.typing.ArrayLike, Callable] = 50. * u.ms,
      beta: Union[bst.typing.ArrayLike, Callable] = 1. * u.mA,
      V_initializer: Callable = bst.init.Constant(0. * u.mV),
      I2_initializer: Callable = bst.init.Constant(0. * u.mA),
      spike_fun: Callable = bst.surrogate.ReluGrad(),
      spk_reset: str = 'soft',
      keep_size: bool = False,
      name: str = None,
      mode: bst.mixin.Mode = None,
  ):
    super().__init__(size, keep_size=keep_size, name=name, mode=mode, spk_fun=spike_fun, spk_reset=spk_reset)

    # params
    self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
    self.R = bst.init.param(R, self.varshape, allow_none=False)
    self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
    self.tau = bst.init.param(tau, self.varshape, allow_none=False)
    self.tau_I2 = bst.init.param(tau_a, self.varshape, allow_none=False)
    self.beta = bst.init.param(beta, self.varshape, allow_none=False)

    # initializers
    self._V_initializer = V_initializer
    self._I_initializer = I2_initializer

  def init_state(self, batch_size=None):
    self.V = bst.ShortTermState(bst.init.param(self._V_initializer, self.varshape, batch_size))
    self.I_adp = bst.ShortTermState(bst.init.param(self._I_initializer, self.varshape, batch_size))

  def update(self, x=0. * u.mA):
    # get last states
    last_spk = self.get_spike()
    last_V = self.V.value - self.V_th_inf * last_spk
    last_I2 = self.I_adp.value - self.beta * last_spk

    # numerical integration for I2
    I2 = last_I2 - last_I2 / self.tau_I2 * bst.environ.get_dt()

    # numerical integration for V
    I_ext = self.sum_current_inputs(last_V, init=x + I2)
    V = last_V + (-last_V + self.V_rest + I_ext * self.R) / self.tau * bst.environ.get_dt()
    V = self.sum_delta_inputs(init=V)

    # update states
    self.I_adp.value = I2
    self.V.value = V
    return self.get_spike(V)

  def get_spike(self, V=None):
    V = self.V.value if V is None else V
    return self.spk_fun((V - self.V_th_inf) / self.V_th_inf)


class SNNNet(bst.Module):
  def save_state(self, **kwargs) -> Dict:
    raise NotImplementedError

  def load_state(self, state_dict: Dict, **kwargs):
    raise NotImplementedError

  def vis_data(self) -> Dict:
    raise NotImplementedError

  def visualize(self, inputs, n2show: int = 5, filename: str = None):
    n_seq = inputs.shape[0]
    indices = np.arange(n_seq)
    batch_size = inputs.shape[1]
    bst.init_states(self, batch_size)

    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
      return self.vis_data()

    res = bst.transform.for_loop(step, indices, inputs, pbar=bst.transform.ProgressBar(10))

    fig, gs = bp.visualize.get_figure(4, n2show, 3., 4.5)
    for i in range(n2show):
      # input spikes
      bp.visualize.raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
      # recurrent spikes
      bp.visualize.raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[2, i])
      ax.plot(indices, res['rec_mem'][:, i].to_decimal(u.mV))
      # output potentials
      ax = fig.add_subplot(gs[3, i])
      ax.plot(indices, res['out'][:, i])

    if filename is None:
      plt.show()
      plt.close()
    else:
      if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
      plt.savefig(filename)
      plt.close()


class SNNCobaNet(SNNNet):
  def __init__(
      self,
      n_in: int,
      n_rec: int,
      n_out: int,
      tau_neu: Union[bst.typing.ArrayLike, Callable] = 10. * u.ms,
      tau_a: Union[bst.typing.ArrayLike, Callable] = 100. * u.ms,
      beta: Union[bst.typing.ArrayLike, Callable] = 1. * u.mA,
      tau_syn: Union[bst.typing.ArrayLike, Callable] = 10. * u.ms,
      tau_out: Union[bst.typing.ArrayLike, Callable] = 10. * u.ms,
      ff_scale: float = 1.,
      rec_scale: float = 1.,
      E_exc: Union[bst.typing.ArrayLike, Callable] = 3. * u.mV,
      E_inh: Union[bst.typing.ArrayLike, Callable] = -3. * u.mV,
      w_ei_ratio: float = 10.,
      conn_method: str = 'dense'
  ):
    super().__init__()

    self.n_exc = int(n_rec * 0.8)
    self.n_inh = n_rec - self.n_exc

    # 1. neurons
    self.pop = GIF(n_rec, tau=tau_neu, tau_a=bst.init.Uniform(tau_a * 0.5, tau_a * 1.5), beta=beta)

    # 2. feedforward
    ff_init = bst.init.KaimingNormal(scale=ff_scale * u.siemens)
    self.ff2r = bst.nn.HalfProjAlignPostMg(
      comm=SignedWLinear(n_in, n_rec, w_init=ff_init),
      syn=Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_exc),
      post=self.pop
    )

    # 3. recurrent
    inh_init = bst.init.KaimingNormal(scale=rec_scale * w_ei_ratio * u.siemens)
    exc_init = bst.init.KaimingNormal(scale=rec_scale * u.siemens)
    if conn_method == 'dense':
      inh2r_conn = SignedWLinear(self.n_inh, n_rec, w_init=inh_init)
      exc2r_conn = SignedWLinear(self.n_exc, n_rec, w_init=exc_init)
    elif conn_method == 'sparse':
      inh2r_conn = CSRLayer(self.n_inh, n_rec, prob=0.1, w_init=inh_init)
      exc2r_conn = CSRLayer(self.n_exc, n_rec, prob=0.1, w_init=exc_init)
    else:
      raise ValueError(f'Unknown connection method: {conn_method}')

    # 3.1 inhibitory-to-recurrent
    self.inh2r = bst.nn.HalfProjAlignPostMg(
      comm=inh2r_conn,
      syn=Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_inh),
      post=self.pop
    )
    # 3.2 excitatory-to-recurrent
    self.exc2r = bst.nn.HalfProjAlignPostMg(
      comm=exc2r_conn,
      syn=Expon.delayed(size=n_rec, tau=tau_syn),
      out=bst.nn.COBA.delayed(E=E_exc),
      post=self.pop
    )

    # 4. output
    self.out = bst.nn.LeakyRateReadout(n_rec, n_out, tau=tau_out)

  def update(self, spk):
    e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.n_exc], axis=-1)
    self.ff2r(spk)
    self.exc2r(e_sps)
    self.inh2r(i_sps)
    return self.out(self.pop())

  def save_state(self, **kwargs) -> Dict:
    return {'ff2r.weight': self.ff2r.comm.weight.value.mantissa,
            'exc2r.weight': self.exc2r.comm.weight.value.mantissa,
            'inh2r.weight': self.inh2r.comm.weight.value.mantissa,
            'out.weight': self.out.weight.value}

  def load_state(self, state_dict: Dict, **kwargs):
    self.ff2r.comm.weight.value.update_mantissa(state_dict['ff2r.weight'])
    self.exc2r.comm.weight.value.update_mantissa(state_dict['exc2r.weight'])
    self.inh2r.comm.weight.value.update_mantissa(state_dict['inh2r.weight'])
    self.out.weight.value = state_dict['out.weight']

  def vis_data(self):
    n_rec = self.pop.num
    return {
      'rec_spk': self.pop.get_spike(),
      'rec_mem': self.pop.V.value[:, np.arange(0, n_rec, n_rec // 10)],
      'out': self.out.r.value,
    }


class Trainer:
  def __init__(
      self,
      target_net: SNNNet,
      optimizer: bst.optim.Optimizer,
      loader: bd.cognitive.TaskLoader,
      args: bst.util.DotDict,
      filepath: str | None = None
  ):
    # the network
    self.target_net = target_net

    # the dataset
    self.loader = loader

    # parameters
    self.args = args
    self.filepath = filepath

    # optimizer
    weights = self.target_net.states().subset(bst.ParamState)
    print(weights)
    self.optimizer = optimizer
    self.optimizer.register_trainable_weights(weights)

  def print(self, msg, file=None):
    if file is not None:
      print(msg, file=file)
    print(msg)

  def _loss(self, out, target):
    # MSE loss
    mse_loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()
    # L1 regularization loss
    l1_loss = 0.
    if self.args.weight_L1 != 0.:
      leaves = self.target_net.states().subset(bst.ParamState).to_dict_values()
      for leaf in leaves:
        l1_loss += self.args.weight_L1 * jnp.sum(jnp.abs(leaf))
    return mse_loss, l1_loss

  def _acc(self, outs, target):
    pred = jnp.argmax(jnp.sum(outs, 0), 1)  # [T, B, N] -> [B, N] -> [B]
    acc = jnp.asarray(pred == target, dtype=bst.environ.dftype()).mean()
    return acc

  @bst.transform.jit(static_argnums=(0,))
  def bptt_train(self, inputs, targets) -> Tuple:
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype())
    indices = jnp.arange(inputs.shape[0])
    bst.init_states(self.target_net, inputs.shape[1])
    weights = self.target_net.states().subset(bst.ParamState)
    warmup = self.args.warmup + inputs.shape[0] if self.args.warmup < 0 else self.args.warmup
    n_sim = int(warmup) if warmup > 0 else 0

    def _step_run(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        out = self.target_net(inp)
      return self._loss(out, targets), out

    def _bptt_grad():
      (mse_losses, l1_losses), outs = bst.transform.for_loop(_step_run, indices, inputs)
      mse_losses = mse_losses[n_sim:].mean()
      l1_losses = l1_losses[n_sim:].mean()
      acc = self._acc(outs[n_sim:], targets)
      return mse_losses + l1_losses, (mse_losses, l1_losses, acc)

    f_grad = bst.transform.grad(_bptt_grad, grad_vars=weights, has_aux=True, return_value=True)
    grads, loss, (mse_losses, l1_losses, acc) = f_grad()
    self.optimizer.update(grads)
    return mse_losses, l1_losses, acc

  def f_sim(self):
    inputs, outputs = next(iter(self.loader))
    inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
    self.target_net.visualize(inputs)

  def f_train(self):
    file = None
    if self.filepath is not None:
      if not os.path.exists(self.filepath):
        os.makedirs(self.filepath)
      file = open(f'{self.filepath}/loss.txt', 'w')
    self.print(self.args, file=file)

    acc_max = 0.
    t0 = time.time()
    for bar_idx, (inputs, outputs) in enumerate(self.loader):
      if bar_idx > gargs.epochs:
        break

      inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
      outputs = jnp.asarray(outputs, dtype=bst.environ.ditype())
      mse_ls, l1_ls, acc = self.bptt_train(inputs, outputs)
      self.optimizer.lr.step_epoch()
      desc = (f'Batch {bar_idx:2d}, '
              f'CE={float(mse_ls):.8f}, '
              f'L1={float(l1_ls):.6f}, '
              f'acc={float(acc):.6f}, '
              f'time={time.time() - t0:.2f} s')
      self.print(desc, file=file)

      if acc > acc_max:
        acc_max = acc
        weights = jax.tree.map(np.asarray, self.target_net.save_state())
        bts.file.msgpack_save(f'{self.filepath}/{bar_idx}/train-results-{bar_idx}.bp', weights)
        self.target_net.visualize(inputs, filename=f'{self.filepath}/{bar_idx}/train-results-{bar_idx}.png')

      t0 = time.time()
      if acc_max > 0.99:
        break
    if file is not None:
      file.close()


def training():
  # filepath
  now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
  if gargs.method == 'bptt':
    filepath = f'results/task-training/{gargs.method}-{gargs.conn_method}-{now}'
  else:
    filepath = (f'results/task-training/'
                f'{gargs.method}-{gargs.diag_jacobian}-{gargs.diag_normalize}-{gargs.conn_method}-{now}')
  # filepath = None

  # data
  task = bd.cognitive.EvidenceAccumulation(dt=gargs.dt, mode='spiking')
  gargs.warmup = -(task.t_recall / gargs.dt)
  loader = TaskLoader(task, batch_size=gargs.batch_size, num_workers=gargs.num_workers)

  # network
  net = SNNCobaNet(
    task.num_inputs,
    gargs.n_rec,
    task.num_outputs,
    beta=gargs.beta * u.mA,
    tau_a=gargs.tau_a * u.ms,
    tau_neu=gargs.tau_neu * u.ms,
    tau_syn=gargs.tau_syn * u.ms,
    tau_out=gargs.tau_out * u.ms,
    ff_scale=gargs.ff_scale,
    rec_scale=gargs.rec_scale,
    w_ei_ratio=gargs.w_ei_ratio,
    conn_method=gargs.conn_method,
  )

  # optimizer
  if gargs.opt == 'adam':
    opt = bst.optim.Adam(lr=gargs.lr)
  elif gargs.opt == 'sgd':
    opt = bst.optim.SGD(lr=gargs.lr)
  else:
    raise ValueError

  # trainer
  trainer = Trainer(net, opt, loader, gargs, filepath=filepath)

  if gargs.mode == 'sim':
    trainer.f_sim()
  else:
    trainer.f_train()


def verification():
  import seaborn as sns

  filepath = r'results\task-training\bptt-dense-2024-09-16 10-40-06'

  def visualize_activity(self, inputs, n2show: int = 4, filename: str = None):
    n_seq = inputs.shape[0]
    indices = np.arange(n_seq)
    batch_size = inputs.shape[1]
    bst.init_states(self, batch_size)

    def step(i, inp):
      with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
        self.update(inp)
        n_exc = int(self.pop.num * 0.8)
        n_inh = self.pop.num - n_exc
        exc_indices = np.arange(0, n_exc, n_exc // 5)
        inh_indices = np.arange(0, n_inh, n_inh // 5) + n_exc
        return {'rec_spk': self.pop.get_spike(),
                'exc_mem': self.pop.V.value[:, exc_indices],
                'inh_mem': self.pop.V.value[:, inh_indices],
                'out': self.out.r.value, }

    res = bst.transform.for_loop(step, indices, inputs, pbar=bst.transform.ProgressBar(10))

    plt.style.use(['science', 'nature', 'notebook'])
    fig, gs = bts.visualize.get_figure(5, n2show, 2.0, 4.)
    for i in range(n2show):
      # input spikes
      bp.visualize.raster_plot(indices, inputs[:, i], ax=fig.add_subplot(gs[0, i]), xlim=(0, n_seq))
      # recurrent spikes
      bp.visualize.raster_plot(indices, res['rec_spk'][:, i], ax=fig.add_subplot(gs[1, i]), xlim=(0, n_seq))
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[2, i])
      ax.plot(indices, u.maybe_decimal(res['exc_mem'][:, i], u.mV))
      plt.ylabel('Excitatory V [mV]')
      # recurrent membrane potentials
      ax = fig.add_subplot(gs[3, i])
      ax.plot(indices, u.maybe_decimal(res['inh_mem'][:, i], u.mV))
      plt.ylabel('Inhibitory V [mV]')
      # output potentials
      ax = fig.add_subplot(gs[4, i])
      ax.plot(indices, res['out'][:, i])

    if filename is None:
      plt.show()
      plt.close()
    else:
      plt.savefig(filename)
      plt.close()

  def visualize_weights(self, show=True):
    if gargs.conn_method == 'dense':
      weights = jnp.abs(jnp.concat([self.exc2r.comm.weight_op.value, self.inh2r.comm.weight_op.value], axis=0))
    else:
      weights = jnp.abs(jnp.concat([self.exc2r.comm.to_dense_conn(), self.inh2r.comm.to_dense_conn()], axis=0))
    weights = np.ma.array(weights, mask=weights == 0)

    fig, gs = bts.visualize.get_figure(1, 1, 5., 5.)
    ax = fig.add_subplot(gs[0, 0])
    # pcolormesh = plt.pcolormesh(weights, cmap='Purples')
    # pcolormesh = plt.pcolormesh(weights, cmap='Reds')
    # pcolormesh = plt.pcolormesh(weights, cmap='seismic')
    pcolormesh = plt.pcolormesh(weights, cmap='cool', vmin=0.0, vmax=1.5)
    cmap = pcolormesh.cmap  # Get the colormap
    cmap.set_bad('white', 1.)  # Set white for NaN values with full alpha
    plt.colorbar(pcolormesh)
    plt.xlabel('To neurons')
    plt.ylabel('From neurons')
    plt.title('Network connectivity')
    if show:
      plt.show()

  def plot_weight_dists(self, show=True, title=''):
    exc_weight = np.abs(self.exc2r.comm.weight.value.mantissa).flatten()
    inh_weight = np.abs(self.inh2r.comm.weight.value.mantissa).flatten()

    fig, gs = bts.visualize.get_figure(1, 2, 3, 3.)
    ax = fig.add_subplot(gs[0, 0])
    bin_res = plt.hist(exc_weight, bins=100, color='blue', alpha=0.7, density=True)
    plt.title('Excitatory weights')
    sns.kdeplot(exc_weight, thresh=0.01)
    plt.xlim(0., bin_res[1].max())

    ax = fig.add_subplot(gs[0, 1])
    bin_res = plt.hist(inh_weight, bins=100, color='blue', alpha=0.7, density=True)
    sns.kdeplot(inh_weight, thresh=0.01)
    plt.title('Inhibitory weights')
    plt.xlim(0., bin_res[1].max())

    if title:
      plt.suptitle(title)

    if show:
      plt.show()

  global gargs
  with open(os.path.join(filepath, 'loss.txt'), 'r') as f:
    line = f.readline().strip().replace('Namespace', 'dict')
    gargs = bst.util.DotDict(eval(line))
    print(gargs)

  bst.environ.set(
    dt=gargs.dt * u.ms,
    mode=bst.mixin.JointMode(bst.mixin.Training(), bst.mixin.Batching())
  )
  task = bd.cognitive.EvidenceAccumulation(dt=gargs.dt, mode='spiking')
  loader = TaskLoader(task, batch_size=gargs.batch_size, num_workers=gargs.num_workers)
  gargs.warmup = -(task.t_recall / gargs.dt)
  net = SNNCobaNet(task.num_inputs,
                   gargs.n_rec,
                   task.num_outputs,
                   beta=gargs.beta * u.mA,
                   tau_a=gargs.tau_a * u.ms,
                   tau_neu=gargs.tau_neu * u.ms,
                   tau_syn=gargs.tau_syn * u.ms,
                   tau_out=gargs.tau_out * u.ms,
                   ff_scale=gargs.ff_scale,
                   rec_scale=gargs.rec_scale,
                   w_ei_ratio=gargs.w_ei_ratio,
                   conn_method=gargs.conn_method, )

  # visualize_weights(net, show=False)
  plot_weight_dists(net, show=False, title='Weight distribution before training')
  weight_data = bts.file.msgpack_load(f'{filepath}/243/train-results-243.bp')
  net.load_state(weight_data)
  # visualize_weights(net)
  plot_weight_dists(net, title='Weight distribution before training')

  inputs, _ = next(iter(loader))
  inputs = jnp.asarray(inputs, dtype=bst.environ.dftype()).transpose(1, 0, 2)
  visualize_activity(net, inputs)


if __name__ == '__main__':
  pass
  training()
  # verification()
