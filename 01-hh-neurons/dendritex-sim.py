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


"""
Implementation of the following models in the paper:

- Li, Guoshi, Craig S. Henriquez, and Flavio Fröhlich. “Unified thalamic model generates
  multiple distinct oscillations with state-dependent entrainment by stimulation.”
  PLoS computational biology 13.10 (2017): e1005797.
"""

import time

import brainstate as bst
import braintools as bts
import brainunit as u
import dendritex as dx
import matplotlib.pyplot as plt


class TRN(dx.neurons.SingleCompartment):
  def __init__(self, size, V_initializer=bst.init.Constant(-70. * u.mV), gl=0.0075):
    super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV)

    self.na = dx.ions.SodiumFixed(size)
    self.na.add_elem(INa=dx.channels.INa_Ba2002(size, V_sh=-40 * u.mV))

    self.k = dx.ions.PotassiumFixed(size, E=-90. * u.mV)
    self.k.add_elem(IDR=dx.channels.IKDR_Ba2002(size, V_sh=-40 * u.mV))
    self.k.add_elem(IKL=dx.channels.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))

    self.ca = dx.ions.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=100. * u.ms, d=0.5 * u.um)
    self.ca.add_elem(ICaN=dx.channels.ICaN_IS2008(size, g_max=0.2 * (u.mS / u.cm ** 2)))
    self.ca.add_elem(ICaT=dx.channels.ICaT_HP1992(size, g_max=1.3 * (u.mS / u.cm ** 2)))

    self.kca = dx.MixIons(self.k, self.ca)
    self.kca.add_elem(IAHP=dx.channels.IAHP_De1994(size, g_max=0.2 * (u.mS / u.cm ** 2)))

    # self.IL = dx.channels.IL(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-60 * u.mV)
    self.IL = dx.channels.IL(size, g_max=gl * (u.mS / u.cm ** 2), E=-60 * u.mV)

  def compute_derivative(self, x=0. * u.nA):
    return super().compute_derivative(x * (1e-3 / (1.43e-4 * u.cm ** 2)))

  def step_run(self, t, inp):
    # dx.rk4_step(neu, t, inp)
    dx.rk2_step(self, t, inp)
    # dx.euler_step(neu, t, inp)
    return self.V.value


def try_trn_neuron():
  import seaborn as sns

  bst.environ.set(dt=0.01 * u.ms)

  I = bts.input.section_input(values=[0, -0.05, 0], durations=[500, 150, 1000], dt=0.01) * u.uA
  times = u.math.arange(I.shape[0]) * bst.environ.get_dt()

  neu = TRN([1, 1], gl=0.0075)  # [n_neuron, n_compartment]
  neu.init_state()

  t0 = time.time()
  vs = bst.transform.for_loop(neu.step_run, times, I)
  t1 = time.time()
  print(f"Elapsed time: {t1 - t0:.4f} s")

  sns.set_theme()
  fig, gs = bts.visualize.get_figure(1, 1, 3, 3)
  fig.add_subplot(gs[0, 0])
  plt.plot(times.to_decimal(u.ms), u.math.squeeze(vs.to_decimal(u.mV)))
  plt.title("BrainPy Simulation")
  plt.ylabel("Membrane Potential (mV)")
  plt.xlim([600, 1200])
  plt.xlabel("Time (ms)")
  plt.gca().yaxis.grid(False)
  plt.show()


if __name__ == '__main__':
  try_trn_neuron()
