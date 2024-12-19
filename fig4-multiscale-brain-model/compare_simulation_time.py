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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'

import jax
jax.config.update('jax_platform_name', 'cpu')

import braintools
import brainunit as u
import jax
import numpy as np
import pandas as pd
import time

import brainstate as bst
from _large_scale_COBA_EI_net_with_unit import create_model as create_model_with_units
from _large_scale_COBA_EI_net_without_unit import create_model as create_model_without_units


def evaluate_compilation_times(scale: float = 1.0, n_run: int = 10):
    g_max = np.asarray([0.10108301, 0.60604239, -0.65, -0.33540355, 0.06]) * u.siemens

    with bst.environ.context(dt=0.1 * u.ms):
        model_with_unit = create_model_with_units(g_max, scale=scale)

        # run the model
        v1_inputs = braintools.input.section_input([0. * u.mA, ], [700. * u.ms, ])
        bg_inputs = u.math.ones_like(v1_inputs) * 10.3
        run_indices = np.arange(v1_inputs.size)

    with bst.environ.context(dt=0.1):
        model_without_unit = create_model_without_units(g_max.mantissa, scale=scale)

    @bst.compile.jit
    def run_with_unit():
        bst.nn.reset_all_states(model_with_unit)
        with bst.environ.context(dt=0.1 * u.ms):
            outs = bst.compile.for_loop(model_with_unit.step_run,
                                        run_indices,
                                        bg_inputs,
                                        v1_inputs)
            return model_with_unit.areas['V1'].E.V.value[0]

    @bst.compile.jit
    def run_without_unit():
        bst.nn.reset_all_states(model_without_unit)
        with bst.environ.context(dt=0.1):
            outs = bst.compile.for_loop(model_without_unit.step_run,
                                        run_indices,
                                        u.get_mantissa(bg_inputs),
                                        u.get_mantissa(v1_inputs))
            return model_without_unit.areas[0].E.V.value[0]

    results = []
    jax.block_until_ready(run_with_unit())  # compile once
    jax.block_until_ready(run_without_unit())  # compile once
    for _ in range(n_run):
        t0 = time.time()
        jax.block_until_ready(run_with_unit())
        t1 = time.time()
        results.append([model_with_unit.num, 'with unit', t1 - t0])
        print(f"scale: {scale}, with unit: {t1 - t0} s")

        t0 = time.time()
        jax.block_until_ready(run_without_unit())
        t1 = time.time()
        results.append([model_without_unit.num, 'without unit', t1 - t0])
        print(f"scale: {scale}, without unit: {t1 - t0} s")

    return results


def with_scales():
    heads = ['n', 'unit_or_not', 'time']
    results = []
    for scale in [1., 5., 10., 50., 100.]:
        results.extend(evaluate_compilation_times(scale=scale))

        platform = jax.default_backend()
        pd.DataFrame(results, columns=heads).to_csv(f'results/{platform}_simulate_time.csv', index=False)


if __name__ == '__main__':
    with_scales()
