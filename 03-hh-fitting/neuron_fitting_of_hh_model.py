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
import time
from typing import Union, Callable

import brainstate as bst
import braintools as bts
import brainunit as u
import dendritex as dx
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bst.environ.set(dt=0.01 * u.ms)

# Load Input and Output Data
df_inp_traces = pd.read_csv('neuron_data/input_traces_hh.csv')
df_out_traces = pd.read_csv('neuron_data/output_traces_hh.csv')

area = 20000 * u.um ** 2  # neuronal area
inp_traces = df_inp_traces.to_numpy()[:, 1:] * 1e9 * u.nA  # input currents
mem_traces = df_out_traces.to_numpy()[:, 1:] * u.mV  # membrane potentials to record
target_vs = u.math.expand_dims(mem_traces.T, axis=-1)  # [T, B, 1]


class INa(dx.Channel):
  root_type = dx.HHTypedNeuron

  def __init__(
      self,
      size: bst.typing.Size,
      ENa: Union[bst.typing.ArrayLike, Callable] = 50. * u.mV,
      gNa: Union[bst.typing.ArrayLike, Callable] = 120. * u.mS,
      vth: Union[bst.typing.ArrayLike, Callable] = -63 * u.mV,
  ):
    super().__init__(size)
    self.ENa = bst.init.param(ENa, self.varshape)
    self.gNa = bst.init.param(gNa, self.varshape)
    self.V_th = bst.init.param(vth, self.varshape)

  def init_state(self, V, batch_size=None):
    self.m = dx.State4Integral(bst.init.param(u.math.zeros, self.varshape))
    self.h = dx.State4Integral(bst.init.param(u.math.zeros, self.varshape))

  #  m channel
  m_alpha = lambda self, V: 0.32 * 4 / u.math.exprel((13. * u.mV - V + self.V_th).to_decimal(u.mV) / 4.)
  m_beta = lambda self, V: 0.28 * 5 / u.math.exprel((V - self.V_th - 40. * u.mV).to_decimal(u.mV) / 5.)
  m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

  # h channel
  h_alpha = lambda self, V: 0.128 * u.math.exprel((17. * u.mV - V + self.V_th).to_decimal(u.mV) / 18.)
  h_beta = lambda self, V: 4. / (1 + u.math.exp((40. * u.mV - V + self.V_th).to_decimal(u.mV) / 5.))
  h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))

  def compute_derivative(self, V, *args, **kwargs):
    m = self.m.value
    h = self.h.value
    self.m.derivative = (self.m_alpha(V) * (1 - m) - self.m_beta(V) * m) / u.ms
    self.h.derivative = (self.h_alpha(V) * (1 - h) - self.h_beta(V) * h) / u.ms

  def current(self, V, *args, **kwargs):
    m = self.m.value
    h = self.h.value
    return (self.gNa * m * m * m * h) * (self.ENa - V)


class IK(dx.Channel):
  root_type = dx.HHTypedNeuron

  def __init__(
      self,
      size: bst.typing.Size,
      EK: Union[bst.typing.ArrayLike, Callable] = -90. * u.mV,
      gK: Union[bst.typing.ArrayLike, Callable] = 36. * u.mS,
      vth: Union[bst.typing.ArrayLike, Callable] = -63 * u.mV,
  ):
    super().__init__(size)
    self.EK = bst.init.param(EK, self.varshape)
    self.gK = bst.init.param(gK, self.varshape)
    self.V_th = bst.init.param(vth, self.varshape)

  def init_state(self, V, batch_size=None):
    self.n = dx.State4Integral(bst.init.param(u.math.zeros, self.varshape))

  # n channel
  n_alpha = lambda self, V: 0.032 * 5 / u.math.exprel((15. * u.mV - V + self.V_th).to_decimal(u.mV) / 5.)
  n_beta = lambda self, V: .5 * u.math.exp((10. * u.mV - V + self.V_th).to_decimal(u.mV) / 40.)
  n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))

  def compute_derivative(self, V, *args, **kwargs):
    n = self.n.value
    self.n.derivative = (self.n_alpha(V) * (1 - n) - self.n_beta(V) * n) / u.ms

  def current(self, V, *args, **kwargs):
    n2 = self.n.value ** 2
    return (self.gK * n2 * n2) * (self.EK - V)


class HH(dx.neurons.SingleCompartment):
  def __init__(
      self,
      size,
      v_initializer: Callable = bst.init.Uniform(-70 * u.mV, -60. * u.mV),
      gL: Union[bst.typing.ArrayLike, Callable] = 0.003 * u.mS,
      gNa: Union[bst.typing.ArrayLike, Callable] = 120. * u.mS,
      gK: Union[bst.typing.ArrayLike, Callable] = 36. * u.mS,
      C: Union[bst.typing.ArrayLike, Callable] = 1. * (u.uF / u.cm ** 2)
  ):
    super().__init__(size, V_initializer=v_initializer, C=C)
    self.ina = INa(size, gNa=gNa)
    self.ik = IK(size, gK=gK)
    self.il = dx.channels.IL(size, g_max=gL, E=-65. * u.mV)


def visualize_target(voltages):
  fig, gs = bts.visualize.get_figure(2, voltages.shape[1], 3, 4.5)
  times = np.arange(voltages.shape[0]) * 0.01
  for i in range(voltages.shape[1]):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(times, voltages.mantissa[:, i], label='target')
    plt.xlabel('Time [ms]')
    plt.legend()
    ax = plt.subplot(gs[1, i])
    ax.plot(times, inp_traces[i].mantissa)
    plt.xlabel('Time [ms]')
  plt.show()


def visualize(voltages, gl, g_na, g_kd, C):
  # currents: [T, B]
  # voltages: [T, B]
  simulated_vs = simulate_model(gl, g_na, g_kd, C)
  voltages = voltages.mantissa
  simulated_vs = simulated_vs.mantissa

  times = np.arange(voltages.shape[0]) * 0.01
  fig, gs = bts.visualize.get_figure(2, simulated_vs.shape[1], 3, 4.5)
  for i in range(simulated_vs.shape[1]):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(times, voltages[:, i], label='target')
    ax.plot(times, simulated_vs[:, i], label='simulated')
    if i == 0:
      plt.ylabel('Voltage [mV]')
    plt.legend()
    ax2 = plt.subplot(gs[1, i])
    ax2.plot(times, inp_traces[i].mantissa)
    if i == 0:
      plt.ylabel('Current [nA]')
    plt.xlabel('Time [ms]')
    fig.align_ylabels([ax, ax2])
  plt.show()


@bst.transform.jit
def simulate_model(gl, g_na, g_kd, C):
  current = inp_traces.T
  assert current.ndim == 2  # [T, B]
  n_input = current.shape[1]
  hh = HH((n_input, 1), gL=gl, gNa=g_na, gK=g_kd, C=C, v_initializer=bst.init.Constant(-65. * u.mV), )
  hh.init_state()

  def step_fun(i, inp):
    with bst.environ.context(i=i, t=bst.environ.get_dt() * i):
      dx.rk2_step(hh, bst.environ.get('t'), inp)
    return hh.V.value

  indices = np.arange(current.shape[0])
  current = u.math.expand_dims(current, axis=-1)  # [T, B, 1]
  return bst.transform.for_loop(step_fun, indices, current)  # (T, B)


@bst.transform.jit
def compare_potentials(param):
  vs = simulate_model(param['gl'], param['g_na'], param['g_kd'], param['C'])  # (T, B)
  vs = vs.in_unit(target_vs.unit)
  losses = bts.metric.squared_error(vs.mantissa, target_vs.mantissa)
  return losses.mean()


# inp_traces: [B, T]
# target_vs: [B, T, 1]
# target_vs = simulate_model(50.37195496 * u.nsiemens,
#                            28.01317016 * u.usiemens,
#                            8.15968937 * u.usiemens,
#                            195.2173985 * u.pfarad)

# visualize_target(target_vs)


bounds = {
  'gl': [1e0, 1e2] * u.nS,
  'g_na': [1e0, 2e2] * u.uS,
  'g_kd': [1e0, 1e2] * u.uS,
  'C': [0.1, 2] * u.uF * u.cm ** -2 * area,
}


def fitting_by_others(method='DE', n_sample=200, n_iter=20):
  print(f"Method: {method}, n_sample: {n_sample}")

  @jax.jit
  @jax.vmap
  @jax.jit
  def loss_with_multiple_run(**params):
    return compare_potentials(params)

  opt = bts.optim.NevergradOptimizer(
    loss_with_multiple_run,
    n_sample=n_sample,
    bounds=bounds,
    method=method,
  )

  times, losses = [], []
  for _ in range(10):
    t0 = time.time()
    param = opt.minimize(n_iter, verbose=False)
    times.append(time.time() - t0)
    loss = compare_potentials(param)
    print(f'Time elapsed: {times[-1]} s, Loss: {loss}')
    print(param)
    # print(loss)
    losses.append(loss)

  print(np.asarray(times).tolist())
  print(np.asarray(losses).tolist())

  # visualize(target_vs, **param)
  # return param, loss


def compare_fitting_with_brian2():
  import seaborn as sns

  # {'Cm': 194.26783035 * pfarad, 'g_na': 29.22276093 * usiemens, 'g_kd': 7.98791554 * usiemens, 'gl': 44.15358841 * nsiemens}

  brian2_data = dict(
    C=194.26783035 * u.pF,
    g_na=29.22276093 * u.uS,
    g_kd=7.98791554 * u.uS,
    gl=44.15358841 * u.nS
  )
  ex_data = {
    'gl': 10.553969 * u.nsiemens,
    'g_na': 26.393143 * u.usiemens,
    'g_kd': 6.557591 * u.usiemens,
    'C': 200.9851 * u.pfarad
  }

  sns.set_theme()

  target_data = target_vs.mantissa
  brian2_sim = simulate_model(**brian2_data).mantissa
  dx_sim = simulate_model(**ex_data).mantissa

  times = np.arange(target_data.shape[0]) * 0.01
  fig, gs = bts.visualize.get_figure(5, brian2_sim.shape[1], 1.5, 3.5)
  for i in range(brian2_sim.shape[1]):
    ax = fig.add_subplot(gs[0:2, i])
    ax.plot(times, target_data[:, i], label='target')
    ax.plot(times, brian2_sim[:, i], label='simulated')
    if i == 0:
      plt.ylabel('Voltage [mV]')
    plt.legend()

    ax2 = fig.add_subplot(gs[2:4, i])
    ax2.plot(times, target_data[:, i], label='target')
    ax2.plot(times, dx_sim[:, i], label='simulated')
    if i == 0:
      plt.ylabel('Voltage [mV]')
    plt.legend()

    ax3 = fig.add_subplot(gs[4, i])
    ax3.plot(times, inp_traces[i].mantissa)
    if i == 0:
      plt.ylabel('Current [nA]')
    plt.xlabel('Time [ms]')
    fig.align_ylabels([ax, ax2, ax3])
  plt.show()


def compare_fitting_time():
  brainpy_times = [2.1433796882629395, 1.849912166595459, 1.5666790008544922, 1.5960915088653564, 1.4829940795898438,
                   1.6175978183746338, 1.4936435222625732, 1.624887228012085, 1.5235929489135742, 1.6421682834625244]
  brainpy_loss = [10.230628967285156, 11.252373695373535, 13.675355911254883, 11.752726554870605, 14.449509620666504,
                  4.639727592468262, 8.56899642944336, 16.59523582458496, 15.274344444274902, 38.004234313964844]

  brian2_times = [1.2727279663085938, 1.3442201614379883, 1.3161647319793701, 1.2976181507110596, 1.3576574325561523,
                  1.3943753242492676, 1.4781625270843506, 1.4289195537567139, 1.4376657009124756, 1.4104557037353516]
  brian2_loss = [32.61633606518837, 26.548679372842116, 22.081774244261734, 21.796113909182512, 19.346898719857485,
                 37.014947127483204, 5.921564615858741, 11.965199890681827, 4.3534883339167205, 34.57297647087796]

  import seaborn as sns
  sns.set_theme(font_scale=1.2)
  fig, gs = bts.visualize.get_figure(1, 2, 3, 4)
  ax = fig.add_subplot(gs[0, 0])
  ax.boxplot(
    [brainpy_times, brian2_times],
    patch_artist=True,  # fill with color
    labels=['Dendritex', 'brian2modelfitting']  # will be used to label x-ticks
  )
  plt.ylabel("Fitting Time [s]")
  ax = fig.add_subplot(gs[0, 1])
  ax.boxplot(
    [brainpy_loss, brian2_loss],
    patch_artist=True,  # fill with color
    labels=['Dendritex', 'brian2modelfitting']  # will be used to label x-ticks
  )
  plt.ylabel("Fitting Loss [mV$^2$]")
  plt.show()


if __name__ == '__main__':
  # fitting_by_others(n_sample=40)
  # compare_fitting_with_brian2()
  compare_fitting_time()
