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

import pickle
import time
from typing import Optional, Callable

import brainstate as bst
import braintools
import brainevent.nn
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


def pop_expon_syn(pre, post, delay, prob, g_max, tau, label=None):
    return bst.nn.AlignPostProj(
        (
            pre.prefetch('spike')
            if (delay is None) or (delay < bst.environ.get_dt()) else
            pre.prefetch('spike').delay.at(delay)
        ),
        comm=brainevent.nn.FixedProb(pre.in_size, post.in_size, prob, g_max),
        syn=bst.nn.Expon.desc(post.in_size, tau=tau, g_initializer=bst.init.ZeroInit(unit=u.siemens)),
        out=bst.nn.CUBA.desc(scale=u.mV),
        post=post,
        label=label
    )


def area_expon_syns(pre, post, delay, prob, gEE=0.03 * u.siemens, tau=5. * u.ms, label=None):
    return pop_expon_syn(pre.E, post.E, delay, prob, g_max=gEE, tau=tau, label=label)


class EINet(bst.nn.Module):
    def __init__(
        self,
        num_E,
        num_I,
        gEE=0.03 * u.siemens,
        gEI=0.03 * u.siemens,
        gIE=0.335 * u.siemens,
        gII=0.335 * u.siemens,
        p=0.2,
        separate_ei: bool = False
    ):
        super().__init__()
        self.E = LifRefLTC(
            num_E,
            tau_ref=5. * u.ms,
            tau=20. * u.ms,
            V_rest=-60. * u.mV,
            V_th=-50. * u.mV,
            V_reset=-60. * u.mV,
            V_initializer=bst.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.I = LifRefLTC(
            num_I,
            tau_ref=5. * u.ms,
            tau=20. * u.ms,
            V_rest=-60. * u.mV,
            V_th=-50. * u.mV,
            V_reset=-60. * u.mV,
            V_initializer=bst.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.E2E = pop_expon_syn(
            self.E,
            self.E,
            None,
            prob=p,
            g_max=gEE,
            tau=5. * u.ms,
            label='intra-E' if separate_ei else None
        )
        self.E2I = pop_expon_syn(
            self.E,
            self.I,
            None,
            prob=p,
            g_max=gEI,
            tau=5. * u.ms,
            label='intra-E' if separate_ei else None
        )
        self.I2E = pop_expon_syn(
            self.I,
            self.E,
            None,
            prob=p,
            g_max=gIE,
            tau=10. * u.ms,
            label='intra-I' if separate_ei else None
        )
        self.I2I = pop_expon_syn(
            self.I,
            self.I,
            None,
            prob=p,
            g_max=gII,
            tau=10. * u.ms,
            label='intra-I' if separate_ei else None
        )

    def update(self, E_bg, I_bg):
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E(E_bg)
        self.I(I_bg)


class VisualSystem(bst.nn.Module):
    def __init__(
        self,
        ne: int,
        ni: int,
        scale: float,
        conn_prob_mat: np.ndarray,
        delay_mat: np.ndarray,
        area_names: list[str],
        p: float = 0.1,
        gEE=0.03 * u.siemens,
        gEI=0.03 * u.siemens,
        gIE=0.335 * u.siemens,
        gII=0.335 * u.siemens,
        muEE=.0375 * u.siemens,
        mon_current=False
    ):
        super().__init__()
        num_area = conn_prob_mat.shape[0]
        ne = int(ne * scale)
        ni = int(ni * scale)

        # brain areas
        self.areas = dict()
        for i, area_name in enumerate(area_names):
            print(f'Building area {area_name} ...')
            self.areas[area_name] = EINet(
                ne,
                ni,
                gEE=gEE,
                gEI=gEI,
                gII=gII,
                gIE=gIE,
                p=p / scale,
                separate_ei=mon_current,
            )

        # projections
        self.projections = dict()
        for i, pre in enumerate(area_names):
            for j, post in enumerate(area_names):
                # if conn_prob_mat[j, i] / scale * self.areas[j].in_size[0] > 0:
                if (conn_prob_mat[j, i] / scale * self.areas[post].E.in_size[0]) >= 1:
                    print(f'Building projection from {pre} to {post} ...')
                    proj = area_expon_syns(
                        self.areas[pre],
                        self.areas[post],
                        delay_mat[j, i],
                        conn_prob_mat[j, i] / scale,
                        gEE=muEE,
                        label=f'{pre}-to-{post}' if mon_current else None
                    )
                    self.projections[f'{pre}-to-{post}'] = proj

        self.mon_current = mon_current

    def get_area_net_input(self, area: str):
        net = self.areas[area]
        e_ei = net.E.sum_current_inputs(0. * u.mA, net.E.V.value, label='intra-E')
        e_ii = net.E.sum_current_inputs(0. * u.mA, net.E.V.value, label='intra-I')
        i_ei = net.I.sum_current_inputs(0. * u.mA, net.I.V.value, label='intra-E')
        i_ii = net.I.sum_current_inputs(0. * u.mA, net.I.V.value, label='intra-I')
        return {
            'E-Ecurrent': e_ei,
            'E-Icurrent': e_ii,
            'I-Ecurrent': i_ei,
            'I-Icurrent': i_ii
        }

    def update(self, bg, v1_bg):
        # call all projections, generating synaptic currents
        for proj in self.projections.values():
            proj()

        # call all areas, updating neurons
        self.areas['V1'](bg + v1_bg, bg)  # V1
        for area in area_names[1:]:  # other areas
            self.areas[area](bg, bg)

        # collect outputs
        outs = {
            'E-sps': u.math.concatenate([area.E.spike.value for area in self.areas.values()]),
            'I-sps': u.math.concatenate([area.I.spike.value for area in self.areas.values()])
        }

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


def create_model(g_max, scale: float = 1.0, mon_current: bool = False):
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
        num_exc,
        num_inh,
        scale,  # scale
        area_names=area_names,
        conn_prob_mat=conn,
        delay_mat=delayMat,
        gEE=g_max[0],
        gEI=g_max[1],
        gIE=g_max[2],
        gII=g_max[3],
        muEE=g_max[4],
        p=0.1,
        mon_current=mon_current,
    )
    bst.nn.init_all_states(model)
    return model


def show_current_components(outs, indices, area):
    # show the raster plot
    times = indices * bst.environ.get_dt()
    sps_show(area_names, outs['E-sps'], times, num_exc, 'E spikes')
    sps_show(area_names, outs['I-sps'], times, num_inh, 'I spikes')

    for i in range(5):
        fig, gs = braintools.visualize.get_figure(4, 1, 1.5, 6.)
        fig.add_subplot(gs[0, 0])
        plt.plot(times, outs[f'{area}-membrane'][..., i])
        fig.add_subplot(gs[1, 0])
        plt.plot(times, outs[f'{area}-syn-g'][..., i])
        fig.add_subplot(gs[2, 0])
        plt.plot(times, outs[f'{area}-intra-current']['E-Ecurrent'][..., i], label='E-E')
        plt.plot(times, outs[f'{area}-intra-current']['E-Icurrent'][..., i], label='E-I')
        plt.plot(times, outs[f'{area}-intra-current']['I-Ecurrent'][..., i], label='I-E')
        plt.plot(times, outs[f'{area}-intra-current']['I-Icurrent'][..., i], label='I-I')
        plt.legend()
        ax = fig.add_subplot(gs[3, 0])
        for name in outs[f'{area}-circuit-current']:
            plt.plot(times, outs[f'{area}-circuit-current'][name][..., i], label=name)
        ax.set_yscale('log')
        plt.legend()
    plt.show()


def evaluate_current_components():
    bst.environ.set(dt=0.1 * u.ms)

    area = 'TEpd'

    v1_inputs = braintools.input.section_input(
        [0. * u.mA, 10. * u.mA, 0. * u.mA],
        [200. * u.ms, 25 * u.ms, 500. * u.ms]
    )

    def simulate():
        g_max = np.asarray([0.10108301, 0.60604239, -0.65, -0.33540355, 0.06]) * u.siemens
        model = create_model(g_max, mon_current=True)

        def step_run(i, *args):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
                outs = model.update(*args)
                monitors = dict()
                monitors[f'{area}-intra-current'] = model.get_area_net_input('TEpd')
                monitors[f'{area}-circuit-current'] = dict()
                for pre in area_names:
                    try:
                        net = model.areas[area].E
                        r = net.sum_current_inputs(0. * u.mA, net.V.value, label=f'{pre}-to-{area}')
                        if r.size != 1:
                            monitors[f'{area}-circuit-current'][pre] = r
                    except KeyError:
                        pass
                monitors[f'{area}-syn-g'] = model.areas[area].E2E.syn.g.value
                monitors[f'{area}-membrane'] = model.areas[area].E.V.value

                monitors = jax.tree.map(lambda x: x[:5], monitors)
                return dict(**outs, **monitors)

        indices = np.arange(v1_inputs.size)
        outs = bst.compile.for_loop(
            step_run,
            indices,
            u.math.ones_like(v1_inputs) * 10.3,
            v1_inputs,
            pbar=bst.compile.ProgressBar(100)
        )

        with open('results/current-components-v2.pkl', 'wb') as f:
            pickle.dump(outs, f)

    def visualize():
        with open('results/current-components-v2.pkl', 'rb') as f:
            outs = pickle.load(f)
        show_current_components(outs, np.arange(v1_inputs.size), area)

    # simulate()
    visualize()


def show_hierarchy_spikes_and_currents():
    bst.environ.set(dt=0.1 * u.ms)
    area = 'TEpd'

    # with open('results/current-components-v2.pkl', 'rb') as f:
    with open('results/current-components.pkl', 'rb') as f:
        outs = pickle.load(f)

    sps = outs['E-sps']
    times = np.arange(sps.shape[0]) * bst.environ.get_dt()

    t0 = 180.
    t1 = 360.

    def get_ax(ax):
        ax.set_xlim(t0, t1)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        return ax

    # # recurrent spikes
    # fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 3.5)
    # ax = get_ax(fig.add_subplot(gs[0, 0]))
    # elements = np.where(sps > 0.)
    # indices = elements[1]
    # times_ = times[elements[0]]
    # plt.plot(times_, indices, '.', markersize=5, rasterized=True)
    # plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
    # plt.ylim(0, len(area_names) * num_exc)
    # plt.xlabel('Time [ms]')
    # plt.savefig('vis/hierarchy-spikes.svg')
    # plt.savefig('vis/hierarchy-spikes.pdf')

    for i in range(5):
        axes = []
        fig, gs = braintools.visualize.get_figure(3, 1, 2, 4.)
        ax = get_ax(fig.add_subplot(gs[0, 0]))
        for j in range(5):
            plt.plot(times, outs[f'{area}-membrane'][..., j], label=f'Neuron {j}')
        ax.axes.get_xaxis().set_visible(False)
        plt.legend(fontsize=8)
        axes.append(ax)
        plt.ylabel('Membrane \npotential [mV]')

        # ax = get_ax(fig.add_subplot(gs[1, 0]))
        # plt.plot(times, outs[f'{area}-syn-g'])
        # ax.axes.get_xaxis().set_visible(False)
        # plt.ylabel('Synaptic \nconductance [S]')
        # axes.append(ax)

        ax = get_ax(fig.add_subplot(gs[1, 0]))
        plt.plot(times, outs[f'{area}-intra-current']['E-Ecurrent'][..., i], label='E-E')
        plt.plot(times, outs[f'{area}-intra-current']['E-Icurrent'][..., i], label='E-I')
        plt.plot(times, outs[f'{area}-intra-current']['I-Ecurrent'][..., i], label='I-E')
        plt.plot(times, outs[f'{area}-intra-current']['I-Icurrent'][..., i], label='I-I')
        # plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
        plt.legend(fontsize=8)
        # plt.xticks([])
        ax.axes.get_xaxis().set_visible(False)
        axes.append(ax)
        plt.ylabel('Intra-area \ncurrent [mA]')

        ax = get_ax(fig.add_subplot(gs[2, 0]))
        for name in outs[f'{area}-circuit-current']:
            plt.plot(times, outs[f'{area}-circuit-current'][name][..., i], label=name)
        ax.set_yscale('log')
        # plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
        plt.legend(fontsize=8)
        axes.append(ax)
        plt.ylabel('Inter-area \ncurrent [mA]')
        plt.xlabel('Time [ms]')

        fig.align_ylabels(axes)
        plt.savefig(f'vis/components-{i}.svg')
        plt.savefig(f'vis/components-{i}.pdf')

    plt.show()


def try_large_scale_system(scale: float = 1.0):
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
    g_max = np.asarray([0.10108301, 0.60604239, -0.638, -0.33540355, 0.06]) * u.siemens
    g_max = np.asarray([0.10108301, 0.60604239, -0.65, -0.33540355, 0.06]) * u.siemens

    # g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.075]) * u.siemens
    # g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.065]) * u.siemens
    # g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.06]) * u.siemens
    model = create_model(g_max)

    # run the model
    t0 = time.time()
    duration = 50.
    duration = 20.
    duration = 30.
    duration = 26.
    print('with unit duration:', duration)
    v1_inputs = braintools.input.section_input(
        [0. * u.mA, 10. * u.mA, 0. * u.mA],
        [200. * u.ms, duration * u.ms, 500. * u.ms]
    )
    bg_inp = 10.3
    bg_inputs = u.math.ones_like(v1_inputs) * bg_inp
    run_indices = np.arange(v1_inputs.size)
    outs = bst.compile.for_loop(
        model.step_run,
        run_indices,
        bg_inputs,
        v1_inputs,
        pbar=bst.compile.ProgressBar(100)
    )
    t1 = time.time()
    print(f'Time cost: {t1 - t0:.2f}s')

    # show the raster plot
    run_indices = run_indices * bst.environ.get_dt()
    sps_show(area_names, outs['E-sps'], run_indices, num_exc, 'E spikes')
    sps_show(area_names, outs['I-sps'], run_indices, num_inh, 'I spikes')

    # show the firing rate
    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 6.)
    fig.add_subplot(gs[0, 0])
    plt.plot(run_indices, braintools.metric.firing_rate(outs['E-sps'], 10. * u.ms), label='E')
    plt.plot(run_indices, braintools.metric.firing_rate(outs['I-sps'], 10. * u.ms), label='I')
    plt.legend()
    plt.show()

    np.savez(
        f'results/with-unit-spikes-scale={scale}-gIE={u.get_mantissa(g_max[2]):.5f}'
        f'-muEE={u.get_mantissa(g_max[4]):.5f}-bg_inp={bg_inp}-duration={duration}.npz',
        E_sps=np.asarray(outs['E-sps']),
        I_sps=np.asarray(outs['I-sps']),
        g_max=np.asarray(u.get_mantissa(g_max)),
    )


def evaluate_compile_time():
    bst.environ.set(dt=0.1 * u.ms)

    g_max = np.asarray([0.10108301, 0.60604239, -0.60977116, -0.33540355, 0.075]) * u.siemens
    model = create_model(g_max)
    v1_inputs = braintools.input.section_input([0. * u.mA, ], [100. * u.ms, ])
    bg_inputs = u.math.ones_like(v1_inputs) * 10.5
    run_indices = np.arange(v1_inputs.size)

    graph_def, treefy_states = bst.graph.flatten(model)

    @jax.jit
    def run(treefy_states_):
        model_ = bst.graph.unflatten(graph_def, treefy_states_)
        outs = bst.compile.for_loop(model_.step_run, run_indices, bg_inputs, v1_inputs)
        _, treefy_states_ = bst.graph.flatten(model_)
        return outs, treefy_states_

    t0 = time.time()
    run.lower(treefy_states).compile()
    print('Compile time:', time.time() - t0)


if __name__ == '__main__':
    pass

    # try_large_scale_system()
    # evaluate_current_components()
    # evaluate_compile_time()

    show_hierarchy_spikes_and_currents()
