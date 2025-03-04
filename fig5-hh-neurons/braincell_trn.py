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

import brainstate
import brainunit as u

import braincell


class TRN(braincell.neuron.SingleCompartment):
    def __init__(self, size, V_initializer=brainstate.init.Constant(-70. * u.mV)):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV)

        self.na = braincell.ion.SodiumFixed(size)
        self.na.add_elem(INa=braincell.channel.INa_Ba2002(size, V_sh=-40 * u.mV))

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add_elem(IDR=braincell.channel.IKDR_Ba2002(size, V_sh=-40 * u.mV))
        self.k.add_elem(IKL=braincell.channel.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))

        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=100. * u.ms, d=0.5 * u.um)
        self.ca.add_elem(ICaN=braincell.channel.ICaN_IS2008(size, g_max=0.2 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaT=braincell.channel.ICaT_HP1992(size, g_max=1.3 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add_elem(IAHP=braincell.channel.IAHP_De1994(size, g_max=0.2 * (u.mS / u.cm ** 2)))

        self.IL = braincell.channel.IL(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-60 * u.mV)

    def compute_derivative(self, x=0. * u.nA):
        area = 1e-3 / (1.43e-4 * u.cm ** 2)
        return super().compute_derivative(x * area)


def try_trn_neuron():
    import braintools
    import time
    import matplotlib.pyplot as plt

    brainstate.environ.set(dt=0.01 * u.ms)

    I = braintools.input.section_input(values=[0, -0.05, 0], durations=[500, 150, 1000], dt=0.01) * u.uA
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()

    neu = TRN(1)
    neu.init_state()

    def step_run(t, inp):
        with brainstate.environ.context(t=t):
            neu.update(inp)
            return neu.V.value


    t0 = time.time()
    vs = brainstate.compile.for_loop(step_run, times, I)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.4f} s")

    fig, gs = braintools.visualize.get_figure(1, 1, 3, 3)
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
