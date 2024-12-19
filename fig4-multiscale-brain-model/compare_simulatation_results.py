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


import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import brainstate as bst
from _large_scale_COBA_EI_net_with_unit import create_model as create_model_with_units, area_names, num_exc
from _large_scale_COBA_EI_net_without_unit import create_model as create_model_without_units


def try_large_scale_system(scale: float = 1.0):
    with bst.environ.context(dt=0.1 * u.ms):
        g_max = np.asarray([0.10108301, 0.60604239, -0.65, -0.33540355, 0.06]) * u.siemens

        # run the model
        v1_inputs = braintools.input.section_input(
            [0. * u.mA, 10. * u.mA, 0. * u.mA],
            [200. * u.ms, 26. * u.ms, 500. * u.ms]
        )
        bg_inputs = u.math.ones_like(v1_inputs) * 10.3
        run_indices = np.arange(v1_inputs.size)

        # simulation with physical units
        # bst.random.seed(0)
        model_with_unit = create_model_with_units(g_max, scale=scale)
        outs_with_unit = bst.compile.for_loop(model_with_unit.step_run,
                                              run_indices,
                                              bg_inputs,
                                              v1_inputs,
                                              pbar=bst.compile.ProgressBar(100))

    # simulation without physical units
    with bst.environ.context(dt=0.1):
        # bst.random.seed(0)
        model_without_unit = create_model_without_units(g_max.mantissa, scale=scale)
        outs_without_unit = bst.compile.for_loop(model_without_unit.step_run,
                                                 run_indices,
                                                 u.get_mantissa(bg_inputs),
                                                 u.get_mantissa(v1_inputs),
                                                 pbar=bst.compile.ProgressBar(100))

    def show_spikes(spikes, title='', filename=None):
        t0 = 180.
        t1 = 360.

        fig, gs = braintools.visualize.get_figure(1, 1, 6.0, 4.5)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlim(t0, t1)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        times = np.arange(spikes.shape[0]) * 0.1
        elements = np.where(spikes > 0.)
        indices = elements[1]
        times_ = times[elements[0]]
        plt.plot(times_, indices, '.', markersize=5, rasterized=True)
        plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
        plt.ylim(0, len(area_names) * num_exc)
        plt.xlabel('Time [ms]')
        if title:
            plt.title(title)
        if filename:
            plt.savefig(filename)

    show_spikes(outs_with_unit['E-sps'], title='Simulation with Units', filename='results/with_units.svg')
    show_spikes(outs_without_unit['E-sps'], title='Simulation without Units', filename='results/without_units.svg')
    plt.show()


if __name__ == '__main__':
    try_large_scale_system(1.)
