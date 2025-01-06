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
        V_rest: bst.typing.ArrayLike = 0.,
        V_reset: bst.typing.ArrayLike = -5.,
        V_th: bst.typing.ArrayLike = 20.,
        R: bst.typing.ArrayLike = 1.,
        tau: bst.typing.ArrayLike = 10.,
        V_initializer: Callable = bst.init.Constant(0.),
        tau_ref: bst.typing.ArrayLike = 0.,
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
        self.t_last_spike = bst.ShortTermState(bst.init.param(bst.init.Constant(-1e7), self.varshape, batch_size))

    def update(self, x=0):
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


def pop_expon_syn(pre, post, delay, prob, g_max, tau):
    if bst.environ.get('precision') == 'bf16':
        g_max = u.math.asarray(g_max, dtype=np.float32)
    return bst.nn.AlignPostProj(
        (
            pre.prefetch('spike')
            if (delay is None) or (delay < bst.environ.get_dt()) else
            pre.prefetch('spike').delay.at(delay)
        ),
        lambda x: x != 0.,
        comm=bst.event.FixedProb(pre.in_size, post.in_size, prob, g_max),
        syn=bst.nn.Expon.desc(post.in_size, tau=tau, g_initializer=bst.init.ZeroInit()),
        out=bst.nn.CUBA.desc(scale=1.),
        post=post
    )


def area_expon_syns(pre, post, delay, prob, gEE=0.03, tau=5.):
    return pop_expon_syn(pre.E, post.E, delay, prob, g_max=gEE, tau=tau)


class EINet(bst.nn.Module):
    def __init__(
        self,
        num_E,
        num_I,
        gEE=0.03,
        gEI=0.03,
        gIE=0.335,
        gII=0.335,
        p=0.2
    ):
        super().__init__()
        self.E = LifRefLTC(
            num_E,
            tau_ref=5.,
            tau=20.,
            V_rest=-60.,
            V_th=-50.,
            V_reset=-60.,
            V_initializer=bst.init.Normal(-55., 2.)
        )
        self.I = LifRefLTC(
            num_I,
            tau_ref=5.,
            tau=20.,
            V_rest=-60.,
            V_th=-50.,
            V_reset=-60.,
            V_initializer=bst.init.Normal(-55., 2.)
        )
        self.E2E = pop_expon_syn(self.E, self.E, None, prob=p, g_max=gEE, tau=5.)
        self.E2I = pop_expon_syn(self.E, self.I, None, prob=p, g_max=gEI, tau=5.)
        self.I2E = pop_expon_syn(self.I, self.E, None, prob=p, g_max=gIE, tau=10.)
        self.I2I = pop_expon_syn(self.I, self.I, None, prob=p, g_max=gII, tau=10.)

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
        gEE=0.03,
        gEI=0.03,
        gIE=0.335,
        gII=0.335,
        muEE=.0375
    ):
        super().__init__()
        num_area = conn_prob_mat.shape[0]
        ne = int(ne * scale)
        ni = int(ni * scale)
        self.num = (ne + ni) * num_area

        # brain areas
        self.areas = []
        for i in range(num_area):
            # print(f'Building area {area_names[i]} ...')
            self.areas.append(
                EINet(ne, ni, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE, p=p / scale)
            )

        # projections
        self.projections = []
        for i in range(num_area):
            for j in range(num_area):
                if (conn_prob_mat[j, i] / scale * self.areas[j].E.in_size[0]) >= 1:
                    # print(f'Building projection from {area_names[i]} to {area_names[j]} ...')
                    proj = area_expon_syns(
                        self.areas[i],
                        self.areas[j],
                        delay_mat[j, i],
                        conn_prob_mat[j, i] / scale,
                        gEE=muEE
                    )
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


    def step_run2(self, i, *args):
        with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
            self.update(*args)


def create_model(g_max, scale: float = 1.0):
    # fraction of labeled neurons
    flnMatp = braintools.file.load_matfile('../fig3-mutiscale-brain-network/Joglekar_2018_data/efelenMatpython.mat')
    conn = np.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i

    # Distance
    speed = 3.5  # axonal conduction velocity
    distMatp = braintools.file.load_matfile('../fig3-mutiscale-brain-network/Joglekar_2018_data/subgraphWiring29.mat')
    distMat = distMatp['wiring'].squeeze()  # distances between areas values..
    delayMat = np.asarray(distMat / speed)

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
    )
    bst.nn.init_all_states(model)
    return model

