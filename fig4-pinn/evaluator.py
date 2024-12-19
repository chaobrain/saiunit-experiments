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
import sys

import jax

jax.config.update('jax_platform_name', 'cpu')

from typing import Sequence, Callable
import pandas

from contextlib import contextmanager
import brainstate as bst
import pinnx


@contextmanager
def change_stdout():
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


def eval(
    problem,
    n_point,
    unit: bool = True,
    n_train: int = 15000,
    external_trainable_variables: Sequence[bst.typing.Variable] = None,
    **kwargs
):
    trainer = pinnx.Trainer(
        problem,
        external_trainable_variables=external_trainable_variables
    )
    _, t1 = trainer.compile(bst.optim.Adam(1e-3), measture_train_step_compile_time=True)
    _, t2 = trainer.train(iterations=n_train, measture_train_step_time=True)
    return [
        n_point,
        'with unit' if unit else 'without unit',
        t1,
        t2,
        trainer.train_state.best_loss_train,
        trainer.train_state.best_loss_test
    ]


def scaling_experiments(
    name: str,
    solve_with_unit: Callable[[float], dict],
    solve_without_unit: Callable[[float], dict],
    scales: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    num_exp: int = 10,
):
    platform = jax.default_backend()

    results = []
    heads = ['data point', 'unit_or_not', 'compile_time', 'train_time', 'train_loss', 'test_loss']
    for scale in scales:
        print(f"scale: {scale}")
        for _ in range(num_exp):
            with change_stdout():
                result1 = eval(**solve_with_unit(scale))
                result2 = eval(**solve_without_unit(scale))
            print(result1)
            print(result2)
            results.append(result1)
            results.append(result2)
        print()

        os.makedirs('results/', exist_ok=True)
        df = pandas.DataFrame(results, columns=heads).to_csv(
            f'results/{name}_{platform}_scaling.csv',
            index=False
        )
