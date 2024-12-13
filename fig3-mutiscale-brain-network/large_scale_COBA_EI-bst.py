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
from typing import Optional, Callable

import brainstate as bst
import braintaichi
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

area_names = [
    'V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1',
    'STPc', '7A', '46d', '10', '9/46v', '9/46d', 'F5', 'TEpd', 'PBr',
    '7m', '7B', 'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c'
]

num_exc = 3200
num_inh = 800


class LifRefLTC(bst.nn.Neuron):
    def __init__(
        self,
        size: bst.typing.Size,
        name: Optional[str] = None,
        V_rest: bst.typing.ArrayLike = 0. * u.mV,
        V_reset: bst.typing.ArrayLike = -5. * u.mV,
        V_th: bst.typing.ArrayLike = 20. * u.mV,
        R: bst.typing.ArrayLike = 1. * u.ohm,
        tau: bst.typing.ArrayLike = 10. * u.ms,
        V_initializer: Callable = bst.init.Constant(0. * u.mV),
        tau_ref: bst.typing.ArrayLike = 0. * u.ms,
    ):
        # initialization
        super().__init__(size, name=name, )

        # parameters
        self.tau_ref = bst.init.param(tau_ref, self.varshape)
        self.V_rest = bst.init.param(V_rest, self.varshape)
        self.V_reset = bst.init.param(V_reset, self.varshape)
        self.V_th = bst.init.param(V_th, self.varshape)
        self.tau = bst.init.param(tau, self.varshape)
        self.R = bst.init.param(R, self.varshape)

        # initializers
        self._V_initializer = V_initializer

    def init_state(self, batch_size=None, **kwargs):
        self.V = bst.ShortTermState(bst.init.param(self._V_initializer, self.varshape, batch_size))
        self.spike = bst.ShortTermState(bst.init.param(u.math.zeros, self.varshape, batch_size))
        self.t_last_spike = bst.ShortTermState(
            bst.init.param(bst.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        )

    def update(self, x=0 * u.mA):
        # integrate membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, self.V.value)) / self.tau
        V = bst.nn.exp_euler_step(dv, self.V.value)
        V = self.sum_delta_inputs(V)

        # refractory
        refractory = (bst.environ.get('t') - self.t_last_spike.value) <= self.tau_ref
        V = u.math.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        spike = V >= self.V_th
        V = u.math.where(spike, self.V_reset, V)
        t_last_spike = u.math.where(spike, bst.environ.get('t'), self.t_last_spike.value)
        self.V.value = V
        self.spike.value = u.math.asarray(spike, dtype=bst.environ.dftype())
        self.t_last_spike.value = t_last_spike
        return spike


class Expon(bst.nn.Synapse, bst.mixin.AlignPost):
    def __init__(
        self,
        size: bst.typing.Size,
        name: Optional[str] = None,
        tau: bst.typing.ArrayLike = 8.0 * u.ms,
    ):
        super().__init__(name=name, in_size=size)
        self.tau = bst.init.param(tau, self.varshape)

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = bst.ShortTermState(
            bst.init.param(bst.init.Constant(0. * u.siemens),
                           self.varshape,
                           batch_size)
        )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g.value = bst.init.param(
            bst.init.Constant(0. * u.siemens), self.varshape, batch_size
        )

    def update(self, x=None):
        self.g.value = self.g.value - self.g.value / self.tau * bst.environ.get_dt()
        if x is not None:
            self.align_post_input_add(x)
        return self.g.value

    def align_post_input_add(self, x):
        self.g.value += x


class EventCSRLinear(bst.nn.Module):
    def __init__(
        self,
        weight,
        prob: float,
        n_pre: int,
        n_post: int,
        seed: int = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.weight = weight
        self.n_pre = n_pre
        self.n_post = n_post
        self.prob = prob
        self.n_conn = int(prob * n_post)
        self.seed = np.random.randint(0, int(1e8)) if seed is None else seed

        self.indptr = u.math.arange(n_pre + 1) * self.n_conn
        self.indices = bst.random.randint(0, n_post, (self.n_pre, self.n_conn))
        self.weight = bst.init.param(weight, (self.n_pre * self.n_conn))

    def update(self, x):
        unit = None
        if isinstance(self.weight, u.Quantity):
            w = self.weight.mantissa
            unit = self.weight.unit
        else:
            w = self.weight
        r = braintaichi.jitc_event_mv_prob_homo(
            x, w, self.prob, self.seed,
            shape=(self.n_pre, self.n_post),
            transpose=True,
            outdim_parallel=False
        )
        if unit is not None:
            r = r * unit
        return r


def pop_expon_syn(pre, post, delay, prob, g_max, tau):
    return bst.nn.AlignPostProj(
        (
            pre.prefetch('spike')
            if delay is None else
            pre.prefetch('spike').delay.at(delay)
        ),
        comm=EventCSRLinear(g_max, prob, pre.in_size[0], post.in_size[0]),
        syn=Expon.desc(post.in_size[0], tau=tau),
        out=bst.nn.CUBA.desc(),
        post=post
    )


def area_expon_syns(pre, post, delay, prob, gEE=0.03 * u.siemens, tau=5. * u.ms):
    return pop_expon_syn(pre.E, post.E, delay, prob, g_max=gEE, tau=tau)


class EINet(bst.nn.Module):
    def __init__(self, num_E, num_I, gEE=0.03 * u.siemens, gEI=0.03 * u.siemens,
                 gIE=0.335 * u.siemens, gII=0.335 * u.siemens, p=0.2):
        super().__init__()
        self.E = LifRefLTC(
            num_E,
            tau_ref=5. * u.ms, tau=20. * u.ms,
            V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            V_initializer=bst.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.I = LifRefLTC(num_I,
                           tau_ref=5. * u.ms, tau=20. * u.ms,
                           V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
                           V_initializer=bst.init.Normal(-55. * u.mV, 2. * u.mV)
                           )
        self.E2E = pop_expon_syn(self.E, self.E, None, prob=p, g_max=gEE, tau=5. * u.ms)
        self.E2I = pop_expon_syn(self.E, self.I, None, prob=p, g_max=gEI, tau=5. * u.ms)
        self.I2E = pop_expon_syn(self.I, self.E, None, prob=p, g_max=gIE, tau=10. * u.ms)
        self.I2I = pop_expon_syn(self.I, self.I, None, prob=p, g_max=gII, tau=10. * u.ms)

    def update(self, E_bg, I_bg):
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E(E_bg)
        self.I(I_bg)


class VisualSystem(bst.nn.Module):
    def __init__(
        self, ne, ni, conn_prob_mat, delay_mat, area_names, p=0.1,
        gEE=0.03 * u.siemens, gEI=0.03 * u.siemens,
        gIE=0.335 * u.siemens, gII=0.335 * u.siemens,
        muEE=.0375 * u.siemens
    ):
        super().__init__()
        num_area = conn_prob_mat.shape[0]

        # brain areas
        self.areas = []
        for i in range(num_area):
            print(f'Building area {area_names[i]} ...')
            self.areas.append(EINet(ne, ni, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE, p=p))

        # projections
        self.projections = []
        for i in range(num_area):
            for j in range(num_area):
                if conn_prob_mat[j, i] > 0:
                    print(f'Building projection from {area_names[i]} to {area_names[j]} ...')
                    proj = area_expon_syns(self.areas[i], self.areas[j], delay_mat[j, i], conn_prob_mat[j, i], gEE=muEE)
                    self.projections.append(proj)

    def update(self, bg, v1_bg):
        # call all projections, generating synaptic currents
        [proj() for proj in self.projections]

        # call all areas, updating neurons
        self.areas[0](bg + v1_bg, bg)  # V1
        for area in self.areas[1:]:  # other areas
            area(bg, bg)

        # collect outputs
        outs = {'E-sps': u.math.concatenate([area.E.spike.value for area in self.areas]),
                'I-sps': u.math.concatenate([area.I.spike.value for area in self.areas])}
        return outs

    def step_run(self, i, *args):
        with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
            return self.update(*args)


def sps_show(area_names, sps, run_indices, num_exc, title):
    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.)
    fig.add_subplot(gs[0, 0])

    # get index and time
    elements = np.where(sps > 0.)
    indices = elements[1]
    times = run_indices[elements[0]]
    plt.plot(times, indices, '.', markersize=1)
    plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
    plt.ylim(0, len(area_names) * num_exc)
    plt.xlim(0., run_indices[-1])
    plt.xlabel('Time [ms]')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.title(title)


def _create_model(g_max):
    # fraction of labeled neurons
    flnMatp = braintools.file.load_matfile('Joglekar_2018_data/efelenMatpython.mat')
    conn = np.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i

    # Distance
    speed = 3.5  # axonal conduction velocity
    distMatp = braintools.file.load_matfile('Joglekar_2018_data/subgraphWiring29.mat')
    distMat = distMatp['wiring'].squeeze()  # distances between areas values..
    delayMat = np.asarray(distMat / speed) * u.ms

    # construct the network model
    print(g_max)
    model = VisualSystem(
        num_exc, num_inh, area_names=area_names, conn_prob_mat=conn, delay_mat=delayMat,
        gEE=g_max[0], gEI=g_max[1], gIE=g_max[2], gII=g_max[3], muEE=g_max[4], p=0.1,
    )
    bst.nn.init_all_states(model)
    return model


def try_large_scale_system():
    bst.environ.set(dt=0.1 * u.ms)

    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.08]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.06]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.07]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.075]) * u.siemens
    # g_max = np.asarray([0.10108301, 0.60604239, -0.70977116, -0.33540355, 0.1]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.65, -0.33540355, 0.08]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.645, -0.33540355, 0.08]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.63, -0.33540355, 0.08]) * u.siemens
    # g_max = np.asarray([0.10108301, 0.60604239, -0.635, -0.33540355, 0.08]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.638, -0.33540355, 0.08]) * u.siemens

    # g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.075]) * u.siemens
    model = _create_model(g_max)

    # run the model
    t0 = time.time()
    duration = 100.
    duration = 70.
    v1_inputs = braintools.input.section_input([0. * u.mA, 10. * u.mA, 0. * u.mA],
                                               [200. * u.ms, duration * u.ms, 500. * u.ms])
    bg_inputs = u.math.ones_like(v1_inputs) * 10.5
    run_indices = np.arange(v1_inputs.size)
    outs = bst.compile.for_loop(model.step_run, run_indices, bg_inputs, v1_inputs,
                                pbar=bst.compile.ProgressBar(500))
    t1 = time.time()
    print(f'Time cost: {t1 - t0:.2f}s')

    # show the raster plot
    run_indices = run_indices * bst.environ.get_dt().to_decimal(u.ms)
    sps_show(area_names, outs['E-sps'], run_indices, num_exc, 'E spikes')
    sps_show(area_names, outs['I-sps'], run_indices, num_inh, 'I spikes')

    # show the firing rate
    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.)
    fig.add_subplot(gs[0, 0])
    plt.plot(run_indices, braintools.metric.firing_rate(outs['E-sps'], 10., dt=0.1), label='E')
    plt.plot(run_indices, braintools.metric.firing_rate(outs['I-sps'], 10., dt=0.1), label='I')
    plt.legend()
    plt.show()

    # np.savez(
    #     f'results/spikes-gIE={u.get_mantissa(g_max[2]):.5f}'
    #     f'-muEE={u.get_mantissa(g_max[4]):.5f}-duration={duration}.npz',
    #     E_sps=np.asarray(outs['E-sps']), I_sps=np.asarray(outs['I-sps']),
    #     g_max=np.asarray(u.get_mantissa(g_max)),
    # )


def try2():
    bst.environ.set(dt=0.1 * u.ms)

    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.075]) * u.siemens
    model = _create_model(g_max)
    v1_inputs = braintools.input.section_input([0. * u.mA, ], [100. * u.ms, ])
    bg_inputs = u.math.ones_like(v1_inputs) * 10.5
    run_indices = np.arange(v1_inputs.size)

    @bst.compile.jit
    def run():
        outs = bst.compile.for_loop(model.step_run, run_indices, bg_inputs, v1_inputs)

    # run the model
    t0 = time.time()
    jax.block_until_ready(run())
    t1 = time.time()
    print(f'Time cost: {t1 - t0:.2f}s')

    # run the model
    t0 = time.time()
    jax.block_until_ready(run())
    t1 = time.time()
    print(f'Time cost: {t1 - t0:.2f}s')


if __name__ == '__main__':
    try_large_scale_system()
    # try2()
